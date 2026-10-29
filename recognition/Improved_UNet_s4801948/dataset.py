import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants / file parsing
# --------------------------------------------------------------------------- #
FOLDERS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "val": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}
_NAME_RE = re.compile(r"^(case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii(\.gz)?$")

DEFAULT_NUM_CLASSES = 6  # verify with audit_dataset(); label ids are 0..C-1
# Confirmed by visual label check (check_labels.py)
CLASS_NAMES = {0: "background", 1: "body", 2: "bone", 3: "bladder", 4: "rectum", 5: "prostate"}
BLADDER_ID, RECTUM_ID, PROSTATE_ID = 3, 4, 5
CACHE_HW = (256, 144)  # canvas for the RAM cache; slices are stored top-left aligned


def parse_filename(path):
    """Return (kind, patient, week, slice) from a case_/seg_ filename, or None."""
    m = _NAME_RE.match(Path(path).name)
    if m is None:
        return None
    kind, patient, week, sl = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return kind, patient, week, sl


def _index_folder(folder, kind):
    """Map (patient, week, slice) -> path for one folder."""
    out = {}
    for p in sorted(Path(folder).glob("*.nii*")):
        parsed = parse_filename(p)
        if parsed is None or parsed[0] != kind:
            continue
        out[parsed[1:]] = str(p)
    return out


def _pair_folders(img_dir, seg_dir):
    """Pair image and seg files by key. Returns list of sample dicts."""
    for d in (img_dir, seg_dir):
        if not Path(d).is_dir():
            raise FileNotFoundError(
                f"Folder not found: {d}\n"
                f"Check --root: it must contain the six keras_slices_* folders "
                f"({', '.join(n for pair in FOLDERS.values() for n in pair)}).")
    imgs = _index_folder(img_dir, "case")
    segs = _index_folder(seg_dir, "seg")
    common = sorted(set(imgs) & set(segs))
    if not common:
        raise RuntimeError(
            f"No paired slices found in {img_dir} / {seg_dir} "
            f"({len(imgs)} case_ files, {len(segs)} seg_ files matched the naming pattern).")
    missing_seg = len(imgs) - len(common)
    missing_img = len(segs) - len(common)
    if missing_seg or missing_img:
        print(f"[dataset] WARNING {img_dir}: {missing_seg} images without seg, "
              f"{missing_img} segs without image (dropped).")
    return [dict(image=imgs[k], mask=segs[k], patient=k[0], week=k[1], slice=k[2])
            for k in common]


# --------------------------------------------------------------------------- #
# Leakage-free splitting
# --------------------------------------------------------------------------- #
def patients_of(samples):
    return {s["patient"] for s in samples}


def check_patient_leakage(splits):
    """Return {(split_a, split_b): set(shared patient ids)} for overlapping splits."""
    names = list(splits)
    leaks = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = patients_of(splits[names[i]]) & patients_of(splits[names[j]])
            if shared:
                leaks[(names[i], names[j])] = shared
    return leaks


def assert_no_leakage(splits):
    leaks = check_patient_leakage(splits)
    if leaks:
        msg = "; ".join(f"{a}/{b}: {len(s)} shared patients" for (a, b), s in leaks.items())
        raise AssertionError(f"Patient-level leakage detected: {msg}")


def _split_by_patient(samples, ratios, seed):
    """Shuffle PATIENTS (not slices) and assign whole patients to splits."""
    by_patient = defaultdict(list)
    for s in samples:
        by_patient[s["patient"]].append(s)
    pids = sorted(by_patient)
    random.Random(seed).shuffle(pids)
    n = len(pids)
    n_train = max(1, round(ratios[0] * n))
    n_val = max(1, round(ratios[1] * n))
    groups = {"train": pids[:n_train],
              "val": pids[n_train:n_train + n_val],
              "test": pids[n_train + n_val:]}
    return {k: [s for p in v for s in by_patient[p]] for k, v in groups.items()}


def make_splits(root, use_provided=True, ratios=(0.70, 0.15, 0.15), seed=42):
    """Build train/val/test sample lists with guaranteed patient-level separation.

    use_provided=True : use the supplied folders if they are leakage-free,
                        otherwise fall back to a patient-level re-split.
    use_provided=False: always pool all files and re-split by patient.
    """
    root = Path(root)
    provided = {name: _pair_folders(root / img_d, root / seg_d)
                for name, (img_d, seg_d) in FOLDERS.items()}

    if use_provided:
        leaks = check_patient_leakage(provided)
        if not leaks:
            assert_no_leakage(provided)
            return provided
        print("[dataset] Provided folders share patients "
              f"({ {k: len(v) for k, v in leaks.items()} }); re-splitting by patient.")

    pooled = [s for split in provided.values() for s in split]
    splits = _split_by_patient(pooled, ratios, seed)
    assert_no_leakage(splits)
    return splits


# --------------------------------------------------------------------------- #
# I/O, preprocessing, encoding
# --------------------------------------------------------------------------- #
def load_nifti_2d(path, dtype=np.float32):
    """Load a NIfTI file holding one 2D slice -> (H, W) array (no reorientation,
    so image and mask stay aligned as long as both are loaded this way)."""
    arr = nib.load(path).get_fdata(dtype=dtype)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D slice after squeeze, got shape {arr.shape}")
    return arr


def normalise_slice(img):
    """Per-slice z-score using non-zero (foreground) pixels. img: (H, W) tensor."""
    fg = img[img > 0]
    if fg.numel() < 10:
        fg = img.flatten()
    mean, std = fg.mean(), fg.std()
    return (img - mean) / (std + 1e-6)


def to_one_hot(mask, num_classes):
    """(H, W) or (B, H, W) long -> (C, H, W) or (B, C, H, W) float."""
    oh = F.one_hot(mask.long(), num_classes)
    return oh.permute(2, 0, 1).float() if mask.dim() == 2 else oh.permute(0, 3, 1, 2).float()


def pad_to_multiple(img, mask, multiple=16):
    """Pad bottom/right so H and W are divisible by ``multiple`` (needed for 4 poolings).
    Returns padded tensors; use ``unpad`` with the original (H, W) to undo."""
    h, w = mask.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return img, mask
    img = F.pad(img, (0, pw, 0, ph), mode="constant", value=float(img.min()))
    mask = F.pad(mask, (0, pw, 0, ph), mode="constant", value=0)
    return img, mask


def pad_to_size(img, mask, target_hw):
    """Pad bottom/right to exactly ``target_hw`` (H, W). Returns (img, mask, valid)
    where ``valid`` is a (H, W) bool tensor, True on real (non-padded) pixels.
    Padded image pixels get the slice's minimum (background) value; padded mask
    pixels get class 0 -- so loss/metrics MUST be restricted to ``valid``."""
    h, w = mask.shape[-2:]
    th, tw = target_hw
    if h > th or w > tw:
        raise ValueError(f"Slice {(h, w)} larger than target {tuple(target_hw)}")
    valid = torch.zeros(th, tw, dtype=torch.bool)
    valid[:h, :w] = True
    if (h, w) != (th, tw):
        img = F.pad(img, (0, tw - w, 0, th - h), mode="constant", value=float(img.min()))
        mask = F.pad(mask, (0, tw - w, 0, th - h), mode="constant", value=0)
    return img, mask, valid


def unpad(t, hw):
    """Crop a padded tensor back to the original (H, W)."""
    return t[..., :hw[0], :hw[1]]


# --------------------------------------------------------------------------- #
# Augmentation (torch-only; one shared geometric transform for image + mask)
# --------------------------------------------------------------------------- #
class JointAugment:
    """Small affine (rotation/scale/translation) + intensity jitter + noise.
    No flips: left/right and superior/inferior orientation are anatomically meaningful."""

    def __init__(self, max_rot_deg=10.0, scale=(0.9, 1.1), max_shift=0.05,
                 contrast=(0.9, 1.1), brightness=0.1, noise_std=0.03, p=0.8):
        self.max_rot = max_rot_deg
        self.scale = scale
        self.max_shift = max_shift
        self.contrast = contrast
        self.brightness = brightness
        self.noise_std = noise_std
        self.p = p

    def __call__(self, img, mask):
        """img: (1, H, W) float; mask: (H, W) long."""
        if torch.rand(1).item() > self.p:
            return img, mask
        _, h, w = img.shape
        ang = math.radians((torch.rand(1).item() * 2 - 1) * self.max_rot)
        s = self.scale[0] + torch.rand(1).item() * (self.scale[1] - self.scale[0])
        tx = (torch.rand(1).item() * 2 - 1) * self.max_shift * 2
        ty = (torch.rand(1).item() * 2 - 1) * self.max_shift * 2
        c, sn = math.cos(ang) / s, math.sin(ang) / s
        # Rotation in PIXEL space expressed in normalised coords (handles H != W).
        theta = torch.tensor([[c, -sn * h / w, tx],
                              [sn * w / h, c, ty]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="border", align_corners=False).squeeze(0)
        m = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                          padding_mode="zeros", align_corners=False)
        mask = m.squeeze(0).squeeze(0).round().long()
        # Intensity jitter (image only)
        a = self.contrast[0] + torch.rand(1).item() * (self.contrast[1] - self.contrast[0])
        b = (torch.rand(1).item() * 2 - 1) * self.brightness
        img = img * a + b + torch.randn_like(img) * self.noise_std
        return img, mask


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class HipMRIDataset(Dataset):
    """2D HipMRI slices.

    Item: dict(image=(1,H,W) float, mask=(H,W) long [or (C,H,W) one-hot float],
               valid=(H,W) bool, orig_h, orig_w, patient, week, slice)

    All items are padded (bottom/right) to ``target_hw`` so mixed-width slices
    (256x128 and 256x144) can be batched. ``valid`` marks real pixels: compute the
    loss and metrics only where valid is True, and use ``unpad`` with (orig_h,
    orig_w) to crop predictions back.
    """

    def __init__(self, samples, num_classes=DEFAULT_NUM_CLASSES, augment=None,
                 return_one_hot=False, target_hw=(256, 144), pad_multiple=16, cache=True):
        self.samples = list(samples)
        self.num_classes = num_classes
        self.augment = augment
        self.return_one_hot = return_one_hot
        self.target_hw = tuple(target_hw) if target_hw else None
        self.pad_multiple = pad_multiple  # only used when target_hw is None
        self._use_cache = cache
        self._img_cache = None   # (N,1,H,W) float16, shared memory
        self._mask_cache = None  # (N,H,W) int8
        self._hw_cache = None    # (N,2) original (h, w)

    def __len__(self):
        return len(self.samples)

    def _read_from_disk(self, i):
        s = self.samples[i]
        img = normalise_slice(torch.from_numpy(load_nifti_2d(s["image"])))
        mask = torch.from_numpy(np.rint(load_nifti_2d(s["mask"])).astype(np.int64))
        if img.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {s['image']}: {tuple(img.shape)} vs {tuple(mask.shape)}")
        if mask.min() < 0 or mask.max() >= self.num_classes:
            raise ValueError(f"{s['mask']}: labels in [{int(mask.min())},{int(mask.max())}] "
                             f"outside 0..{self.num_classes - 1}")
        return img, mask

    def preload(self):
        """Fill the RAM cache in the calling process. Stored as three big shared-memory
        tensors (not thousands of small ones) so DataLoader workers started with
        fork/forkserver/spawn all work without exhausting file descriptors."""
        if not self._use_cache or self._img_cache is not None:
            return
        n = len(self.samples)
        H, W = CACHE_HW
        imgs = torch.zeros(n, 1, H, W, dtype=torch.float16)
        masks = torch.zeros(n, H, W, dtype=torch.int8)
        hw = torch.zeros(n, 2, dtype=torch.int32)
        for i in range(n):
            img, mask = self._read_from_disk(i)
            h, w = mask.shape
            if h > H or w > W:
                raise ValueError(f"Slice {(h, w)} exceeds cache canvas {CACHE_HW}")
            imgs[i, 0, :h, :w] = img.half()
            masks[i, :h, :w] = mask.to(torch.int8)
            hw[i, 0], hw[i, 1] = h, w
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"[dataset] preloaded {i + 1}/{n}", flush=True)
        self._img_cache = imgs.share_memory_()
        self._mask_cache = masks.share_memory_()
        self._hw_cache = hw.share_memory_()

    def _load(self, i):
        """Return (img (1,h,w) float32, mask (h,w) long) at the slice's original size."""
        if not self._use_cache:
            img, mask = self._read_from_disk(i)
            return img.unsqueeze(0), mask
        if self._img_cache is None:
            self.preload()
        h, w = int(self._hw_cache[i, 0]), int(self._hw_cache[i, 1])
        return self._img_cache[i, :, :h, :w].float(), self._mask_cache[i, :h, :w].long()

    def __getitem__(self, i):
        img, mask = self._load(i)
        img = img.float()
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        orig_h, orig_w = mask.shape
        if self.target_hw is not None:
            img, mask, valid = pad_to_size(img, mask, self.target_hw)
        else:
            img, mask = pad_to_multiple(img, mask, self.pad_multiple)
            valid = torch.zeros(mask.shape, dtype=torch.bool)
            valid[:orig_h, :orig_w] = True
        s = self.samples[i]
        return dict(image=img,
                    mask=to_one_hot(mask, self.num_classes) if self.return_one_hot else mask,
                    valid=valid, orig_h=orig_h, orig_w=orig_w,
                    patient=s["patient"], week=s["week"], slice=s["slice"])


def get_dataloaders(root, batch_size=16, num_classes=DEFAULT_NUM_CLASSES, augment=True,
                    use_provided=True, ratios=(0.70, 0.15, 0.15), seed=42,
                    num_workers=2, return_one_hot=False, target_hw=(256, 144), cache=True):
    """Return (loaders dict, splits dict). Augmentation is applied to train only."""
    splits = make_splits(root, use_provided=use_provided, ratios=ratios, seed=seed)
    assert_no_leakage(splits)
    loaders = {}
    for name, samples in splits.items():
        ds = HipMRIDataset(samples, num_classes=num_classes,
                           augment=JointAugment() if (augment and name == "train") else None,
                           return_one_hot=return_one_hot, target_hw=target_hw, cache=cache)
        if cache:
            ds.preload()
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                                   num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                   persistent_workers=num_workers > 0,
                                   drop_last=(name == "train"))
    return loaders, splits


# --------------------------------------------------------------------------- #
# Data audit (feeds the feasibility review)
# --------------------------------------------------------------------------- #
def audit_dataset(root, num_classes=DEFAULT_NUM_CLASSES, max_files_per_split=None):
    """Print counts, patient overlap, shapes, alignment, and class statistics."""
    splits = make_splits(root, use_provided=True)
    print("=== Provided-folder leakage check ===")
    provided = {n: _pair_folders(Path(root) / a, Path(root) / b) for n, (a, b) in FOLDERS.items()}
    leaks = check_patient_leakage(provided)
    print("Overlaps:", {k: len(v) for k, v in leaks.items()} if leaks else "none")
    print("Effective split used:", {k: len(v) for k, v in splits.items()})

    for name, samples in splits.items():
        use = samples if max_files_per_split is None else samples[:max_files_per_split]
        px = np.zeros(num_classes, dtype=np.int64)
        present = np.zeros(num_classes, dtype=np.int64)
        shapes, aligned, bad_labels = set(), 0, 0
        for s in use:
            m = np.rint(load_nifti_2d(s["mask"])).astype(np.int64)
            img_nii, seg_nii = nib.load(s["image"]), nib.load(s["mask"])
            shapes.add(m.shape)
            aligned += int(img_nii.shape == seg_nii.shape and
                           np.allclose(img_nii.affine, seg_nii.affine, atol=1e-3))
            if m.min() < 0 or m.max() >= num_classes:
                bad_labels += 1
                m = np.clip(m, 0, num_classes - 1)
            cnt = np.bincount(m.ravel(), minlength=num_classes)
            px += cnt
            present += (cnt > 0)
        n = max(len(use), 1)
        print(f"\n=== {name} : {len(samples)} slices, {len(patients_of(samples))} patients "
              f"(audited {len(use)}) ===")
        print("shapes:", shapes, "| image/mask shape+affine match:", f"{aligned}/{len(use)}",
              "| files with out-of-range labels:", bad_labels)
        print("class | pixel fraction | % slices containing class")
        for c in range(num_classes):
            print(f"  {c}   | {px[c] / px.sum():.4f}        | {100 * present[c] / n:.1f}")
        weeks = sorted({s["week"] for s in samples})
        sl = [s["slice"] for s in samples]
        print("weeks:", weeks, "| slice idx range:", (min(sl), max(sl)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    ap.add_argument("--max_files", type=int, default=None, help="limit files per split for a fast audit")
    args = ap.parse_args()
    audit_dataset(args.root, args.num_classes, args.max_files)
