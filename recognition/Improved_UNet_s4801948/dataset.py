import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

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
    imgs = _index_folder(img_dir, "case")
    segs = _index_folder(seg_dir, "seg")
    common = sorted(set(imgs) & set(segs))
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
