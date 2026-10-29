"""
modules.py -- model components, loss and metrics for 2D HipMRI segmentation.

Pure PyTorch. Contents
---------------------------------
* UNet            : configurable-width/depth 2D U-Net (Ronneberger et al., 2015),
                    with BatchNorm and "same" padding so output size == input size.
* DiceCELoss      : cross-entropy + soft Dice, both restricted to ``valid`` pixels
                    (the padded area added by dataset.py is ignored).
* SegMetrics      : accumulates a confusion matrix over a whole split -> per-class
                    Dice and IoU (NaN where a class is absent in both GT and prediction).
* per_sample_dice : (B, C) per-slice Dice, used for the base/mid/apex analysis (A2).
* leakage_counts  : numerator/denominator for the boundary-leakage criterion (A3).

Conventions: logits (B, C, H, W); target (B, H, W) long; valid (B, H, W) bool.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, with optional dropout between the convs."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """2x2 max-pool followed by DoubleConv."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample (transposed conv or bilinear), concatenate skip, DoubleConv."""

    def __init__(self, in_ch, out_ch, bilinear=False, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(in_ch, in_ch // 2, 1))
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against odd spatial sizes (not an issue for 256x144 / 4 poolings).
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """2D U-Net.

    Args:
        in_channels : input channels (1 for a single MRI slice; 3 for a 2.5D stack).
        num_classes : output channels (logits).
        base        : filters at the first level (64 = original; 32 ~ 4x fewer params).
        depth       : number of down-sampling steps (4 = original; input H, W must be
                      divisible by 2**depth).
        bilinear    : bilinear upsampling instead of transposed convolutions.
        dropout     : Dropout2d probability inside the deepest two levels only.
    """

    def __init__(self, in_channels=1, num_classes=6, base=64, depth=4,
                 bilinear=False, dropout=0.0):
        super().__init__()
        self.depth = depth
        widths = [base * 2 ** i for i in range(depth + 1)]  # e.g. 64,128,256,512,1024
        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            Down(widths[i], widths[i + 1], dropout if i >= depth - 2 else 0.0)
            for i in range(depth))
        self.ups = nn.ModuleList(
            Up(widths[i + 1], widths[i], bilinear, dropout if i >= depth - 2 else 0.0)
            for i in reversed(range(depth)))
        self.outc = nn.Conv2d(widths[0], num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[-2] % (2 ** self.depth) or x.shape[-1] % (2 ** self.depth):
            raise ValueError(f"Input {tuple(x.shape[-2:])} must be divisible by {2 ** self.depth}")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()  # bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class DiceCELoss(nn.Module):
    """ce_weight * CE + dice_weight * (1 - mean soft Dice), over valid pixels only.

    Soft Dice is computed per class over the whole batch (more stable than per-slice
    for classes that appear in few slices), then averaged over ``dice_classes``.

    Args:
        num_classes   : C.
        class_weights : optional (C,) tensor for the CE term (e.g. inverse frequency).
        include_background_in_dice : if False, class 0 is excluded from the Dice mean
                        (so the huge background does not dominate the average).
        smooth        : additive smoothing; an absent class that is also not predicted
                        scores Dice = 1 (no gradient), a false positive on it is penalised.
    """

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, class_weights=None,
                 include_background_in_dice=True, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.first_dice_class = 0 if include_background_in_dice else 1
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, target, valid=None):
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        vf = valid.float()

        ce = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none")
        ce = (ce * vf).sum() / vf.sum().clamp(min=1.0)

        probs = logits.softmax(dim=1) * vf.unsqueeze(1)                      # (B,C,H,W)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float() * vf.unsqueeze(1)
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)             # (C,)
        dice_loss = 1.0 - dice[self.first_dice_class:].mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class SegMetrics:
    """Accumulate a confusion matrix over a split, then report per-class Dice / IoU.

    Usage:
        m = SegMetrics(num_classes); m.update(pred, target, valid) per batch;
        out = m.compute()  # {"dice": (C,), "iou": (C,), "mean_dice": float, ...}
    Classes absent from both GT and prediction give NaN (excluded from means).
    """

    def __init__(self, num_classes):
        self.C = num_classes
        self.reset()

    def reset(self):
        self.conf = torch.zeros(self.C, self.C, dtype=torch.long)  # rows = GT, cols = pred

    @torch.no_grad()
    def update(self, pred, target, valid=None):
        """pred: (B,H,W) long class ids (argmax of logits); target: (B,H,W) long."""
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        t, p = target[valid], pred[valid]
        idx = t * self.C + p
        self.conf += torch.bincount(idx, minlength=self.C ** 2).view(self.C, self.C).cpu()

    def compute(self):
        conf = self.conf.double()
        tp = conf.diag()
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        nan = torch.tensor(float("nan"), dtype=torch.double)
        dice_den, iou_den = 2 * tp + fp + fn, tp + fp + fn
        dice = torch.where(dice_den > 0, 2 * tp / dice_den.clamp(min=1), nan)
        iou = torch.where(iou_den > 0, tp / iou_den.clamp(min=1), nan)
        return {"dice": dice.float(), "iou": iou.float(),
                "mean_dice": torch.nanmean(dice).item(),
                "mean_dice_fg": torch.nanmean(dice[1:]).item(),
                "mean_iou": torch.nanmean(iou).item(),
                "pixel_accuracy": (tp.sum() / conf.sum().clamp(min=1)).item()}


@torch.no_grad()
def per_sample_dice(pred, target, valid, num_classes):
    """Per-slice, per-class Dice -> (B, C). NaN where class is absent in both GT and
    prediction for that slice (so empty slices can be excluded from region averages)."""
    vf = valid.float().unsqueeze(-1)
    p = F.one_hot(pred, num_classes).float() * vf
    t = F.one_hot(target, num_classes).float() * vf
    inter = (p * t).sum((1, 2))
    denom = p.sum((1, 2)) + t.sum((1, 2))
    nan = torch.full_like(denom, float("nan"))
    return torch.where(denom > 0, 2 * inter / denom.clamp(min=1), nan)


@torch.no_grad()
def leakage_counts(pred, target, valid, pred_class, other_classes):
    """Boundary-leakage counts for criterion A3.
    Returns (leaked, predicted): number of predicted ``pred_class`` pixels whose GT is
    in ``other_classes`` (e.g. rectum, bladder), and total predicted ``pred_class`` pixels.
    Accumulate both over the split; leakage = sum(leaked) / sum(predicted)."""
    pc = (pred == pred_class) & valid
    other = torch.zeros_like(target, dtype=torch.bool)
    for c in other_classes:
        other |= target == c
    return (pc & other).sum().item(), pc.sum().item()
