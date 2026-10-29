"""
modules.py -- model components, loss and metrics for 2D HipMRI segmentation.

Contents
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
# Improved U-Net
# --------------------------------------------------------------------------- #
class ConvNormAct(nn.Module):
    """Conv -> InstanceNorm -> LeakyReLU (used in the upsampling/localization modules)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, neg_slope=0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(neg_slope, inplace=True))

    def forward(self, x):
        return self.block(x)


class PreActResBlock(nn.Module):
    """Pre-activation residual 'context module':
        y = conv(act(norm(x)));  y = dropout(y);  y = conv(act(norm(y)));  return x + y
    Normalisation and activation come BEFORE each convolution, so the identity path is
    never normalised or rectified: gradients flow through the skip connection unchanged."""

    def __init__(self, ch, dropout=0.3, neg_slope=0.01):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(neg_slope, inplace=True)

    def forward(self, x):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.act(self.norm2(y)))
        return x + y


class ImprovedUNet(nn.Module):
    """2D Improved U-Net.

    Encoder  : 3x3 stem conv -> context module; each deeper level = stride-2 3x3 conv
               (learned downsampling) -> context module.
    Decoder  : upsampling module (nearest x2 + 3x3 conv halving channels), concatenation with
               the encoder skip, localization module (3x3 conv, then 1x1 conv halving channels).
    Deep supervision: 1x1 segmentation layers on the ``ds_levels`` highest-resolution decoder
               levels; each is upsampled x2 and added element-wise to the next finer level's
               segmentation map, so the final logits are the SUM of multi-scale predictions and
               early/coarse decoder layers receive a direct gradient.

    Args:
        base      : filters at the first level (paper: 16 in 3D; 32 is a good 2D default).
        depth     : number of down-sampling steps; H and W must be divisible by 2**depth.
        dropout   : Dropout2d probability inside every context module (paper: 0.3).
        ds_levels : number of decoder levels with a segmentation head (<= depth; paper: 3).
    forward(x) -> logits (B, C, H, W). With return_aux=True returns (logits, [per-level
    head outputs at native resolution, finest first]) for optional auxiliary losses.
    """

    def __init__(self, in_channels=1, num_classes=6, base=32, depth=4, dropout=0.3,
                 ds_levels=3, neg_slope=0.01):
        super().__init__()
        self.depth = depth
        self.n_heads = min(ds_levels, depth)
        self.neg_slope = neg_slope
        w = [base * 2 ** i for i in range(depth + 1)]

        self.stem = nn.Conv2d(in_channels, w[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList(PreActResBlock(w[i], dropout, neg_slope) for i in range(depth + 1))
        self.down_convs = nn.ModuleList(nn.Conv2d(w[i - 1], w[i], 3, stride=2, padding=1)
                                        for i in range(1, depth + 1))

        # decoder modules ordered from the deepest level (depth-1) up to level 0
        self.up_modules = nn.ModuleList(
            nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                          ConvNormAct(w[i + 1], w[i], 3, neg_slope))
            for i in reversed(range(depth)))
        self.loc_modules = nn.ModuleList(
            nn.Sequential(ConvNormAct(2 * w[i], 2 * w[i], 3, neg_slope),
                          ConvNormAct(2 * w[i], w[i], 1, neg_slope))
            for i in reversed(range(depth)))
        # heads[l] sits on decoder level l (0 = full resolution)
        self.heads = nn.ModuleList(nn.Conv2d(w[l], num_classes, 1) for l in range(self.n_heads))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=self.neg_slope, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_aux=False):
        if x.shape[-2] % (2 ** self.depth) or x.shape[-1] % (2 ** self.depth):
            raise ValueError(f"Input {tuple(x.shape[-2:])} must be divisible by {2 ** self.depth}")
        x = self.enc_blocks[0](self.stem(x))
        skips = [x]
        for i in range(1, self.depth + 1):
            x = self.enc_blocks[i](self.down_convs[i - 1](x))
            if i < self.depth:
                skips.append(x)

        head_out = {}
        for j, level in enumerate(reversed(range(self.depth))):
            x = self.up_modules[j](x)
            x = self.loc_modules[j](torch.cat([skips[level], x], dim=1))
            if level < self.n_heads:
                head_out[level] = self.heads[level](x)

        seg = None  # combine coarse -> fine by upsample + element-wise sum
        for level in reversed(range(self.n_heads)):
            h = head_out[level]
            seg = h if seg is None else F.interpolate(seg, size=h.shape[-2:], mode="bilinear",
                                                      align_corners=False) + h
        if return_aux:
            return seg, [head_out[l] for l in range(self.n_heads)]
        return seg


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
