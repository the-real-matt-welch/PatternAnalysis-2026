"""
predict.py -- example usage of a trained 2D U-Net: load saved weights, run inference on
held-out cases, print results and produce the figures needed for the failure-case autopsy.

Whole split (default: test):
    python predict.py --root /path/to/data --run_dir runs/unet_b64_s0

Single slice:
    python predict.py --run_dir runs/unet_b64_s0 --image path/to/case_XXX_week_X_slice_XX.nii.gz \
                      [--mask path/to/seg_XXX_week_X_slice_XX.nii.gz]

Outputs (in --out_dir, default <run_dir>/predictions/):
    cases_typical.png         median-performance prostate slices
    cases_worst.png           worst prostate slices (failure autopsy)
    cases_false_positive.png  slices with NO ground-truth prostate where one was predicted
    dice_by_position.png      prostate Dice along the gland, base (0) -> apex (1)  [criterion A2]
    confusion_matrix.png      row-normalised pixel confusion (where do classes leak into each other)
    predict_results.json      all printed numbers

Each case row: input | ground truth | prediction | prostate FP/FN overlay | all-class errors.
NumPy is used here for visualisation and bookkeeping (allowed in this script).
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from dataset import (BLADDER_ID, CLASS_NAMES, PROSTATE_ID, RECTUM_ID, HipMRIDataset,
                     load_nifti_2d, make_splits, normalise_slice, pad_to_size)
from modules import SegMetrics, build_model, count_parameters
from train import check_criteria, regional_analysis

CLASS_ALPHA = np.array([0.0, 0.18, 0.5, 0.6, 0.6, 0.7])  # body is faint so it does not hide the anatomy


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def nan_to_none(x):
    return [None if (v is None or v != v) else float(v) for v in x]


# --------------------------------------------------------------------------- #
# Model / inference
# --------------------------------------------------------------------------- #
def load_model(run_dir, ckpt_name, device):
    ck = torch.load(Path(run_dir) / ckpt_name, map_location=device)
    a = ck["args"]
    model = build_model(a.get("model", "unet"), 1, a["num_classes"], base=a["base"],
                        dropout=a.get("dropout", 0.0))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"Loaded {ckpt_name}: epoch {ck['epoch']}, val fg Dice {ck['val_fg_dice']:.4f}, "
          f"base={a['base']}, {count_parameters(model) / 1e6:.2f} M params")
    return model, a["num_classes"], ck


@torch.no_grad()
def run_inference(model, samples, num_classes, device, use_amp, batch_size):
    """Predict every slice in ``samples``; returns arrays (padded size) + metadata."""
    ds = HipMRIDataset(samples, num_classes=num_classes, augment=None, cache=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out, t_inf, n = defaultdict(list), 0.0, 0
    for b in loader:
        x = b["image"].to(device)
        sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
        sync(device)
        t_inf += time.perf_counter() - t0
        n += x.shape[0]
        out["img"].append(b["image"][:, 0].numpy())
        out["gt"].append(b["mask"].numpy())
        out["pred"].append(logits.argmax(1).cpu().numpy())
        out["valid"].append(b["valid"].numpy())
        for k in ("patient", "week", "slice", "orig_h", "orig_w"):
            out[k].append(b[k].numpy())
    data = {k: np.concatenate(v) for k, v in out.items()}
    return data, 1e3 * t_inf / max(n, 1)


# --------------------------------------------------------------------------- #
# Bookkeeping: per-slice records, positions along the gland
# --------------------------------------------------------------------------- #
def build_records(d, cls):
    v, pc, bl = d["valid"], cls["prostate"], cls["bladder"]
    pm, gm = (d["pred"] == pc) & v, (d["gt"] == pc) & v
    cols = ((pm & gm).sum((1, 2)), pm.sum((1, 2)), gm.sum((1, 2)), ((d["gt"] == bl) & v).sum((1, 2)))
    return [dict(patient=int(a), week=int(w), slice=int(s), inter=int(i), pred_p=int(p),
                 gt_p=int(t), gt_bladder=int(g))
            for a, w, s, i, p, t, g in zip(d["patient"], d["week"], d["slice"], *cols)]


def slice_dice(r):
    den = r["pred_p"] + r["gt_p"]
    return 2 * r["inter"] / den if den > 0 else float("nan")


def volume_info(records):
    """(patient, week) -> (first GT-prostate slice, last, high_is_base or None).
    The base is the end of the gland nearer the bladder."""
    vols = defaultdict(list)
    for r in records:
        vols[(r["patient"], r["week"])].append(r)
    info = {}
    for key, rs in vols.items():
        gt_idx = [r["slice"] for r in rs if r["gt_p"] > 0]
        if not gt_idx:
            continue
        lo, hi = min(gt_idx), max(gt_idx)
        bw = sum(r["gt_bladder"] for r in rs)
        high_is_base = None if bw == 0 else (sum(r["slice"] * r["gt_bladder"] for r in rs) / bw) > (lo + hi) / 2
        info[key] = (lo, hi, high_is_base)
    return info


def position(r, info):
    """Return (label, t) where t in [0,1] runs base(0) -> apex(1), or (label, None)."""
    key = (r["patient"], r["week"])
    if key not in info:
        return "no GT prostate in volume", None
    lo, hi, high_is_base = info[key]
    if r["slice"] < lo or r["slice"] > hi:
        return "outside GT span", None
    if hi == lo or high_is_base is None:
        return "in GT span", None
    u = (r["slice"] - lo) / (hi - lo)
    t = 1 - u if high_is_base else u
    return ["base", "mid", "apex"][min(2, int(3 * t))] + " third", t


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_palette(C):
    return np.array(plt.get_cmap("tab10")(np.arange(C)))[:, :3]


def show_mask(ax, img, mask, palette, alphas, vlim):
    ax.imshow(img, cmap="gray", vmin=vlim[0], vmax=vlim[1])
    rgba = np.zeros(mask.shape + (4,))
    rgba[..., :3] = palette[mask]
    rgba[..., 3] = alphas[mask]
    ax.imshow(rgba)


def draw_cases(idx_list, d, records, info, focus, num_classes, path, title):
    if not idx_list:
        print(f"[plot] no cases for '{title}', skipped")
        return
    palette, alphas = make_palette(num_classes), CLASS_ALPHA[:num_classes]
    n = len(idx_list)
    fig, ax = plt.subplots(n, 5, figsize=(13, 3.3 * n), squeeze=False)
    for row, k in enumerate(idx_list):
        h, w = int(d["orig_h"][k]), int(d["orig_w"][k])
        img, gt, pr = d["img"][k, :h, :w], d["gt"][k, :h, :w], d["pred"][k, :h, :w]
        vlim = np.percentile(img, [1, 99.5])
        r = records[k]
        pos, _ = position(r, info)
        dsc = slice_dice(r)
        ax[row, 0].imshow(img, cmap="gray", vmin=vlim[0], vmax=vlim[1])
        ax[row, 0].set_title(f"P{r['patient']} wk{r['week']} slice {r['slice']}\n{pos}", fontsize=8)
        show_mask(ax[row, 1], img, gt, palette, alphas, vlim)
        ax[row, 1].set_title(f"ground truth (prostate {r['gt_p']} px)", fontsize=8)
        show_mask(ax[row, 2], img, pr, palette, alphas, vlim)
        ax[row, 2].set_title(f"prediction (prostate {r['pred_p']} px)", fontsize=8)
        # prostate FP / FN overlay
        fp, fn, tp = (pr == focus) & (gt != focus), (gt == focus) & (pr != focus), (gt == focus) & (pr == focus)
        rgba = np.zeros(gt.shape + (4,))
        rgba[tp] = (0.0, 0.8, 0.0, 0.35)
        rgba[fp] = (1.0, 0.0, 0.0, 0.85)
        rgba[fn] = (0.1, 0.3, 1.0, 0.85)
        ax[row, 3].imshow(img, cmap="gray", vmin=vlim[0], vmax=vlim[1])
        ax[row, 3].imshow(rgba)
        ax[row, 3].set_title(f"{CLASS_NAMES[focus]}: Dice {dsc:.2f}" if dsc == dsc else
                             f"{CLASS_NAMES[focus]}: empty/empty", fontsize=8)
        # all-class error map
        err = np.zeros(gt.shape + (4,))
        err[gt != pr] = (1.0, 0.85, 0.0, 0.85)
        ax[row, 4].imshow(img, cmap="gray", vmin=vlim[0], vmax=vlim[1])
        ax[row, 4].imshow(err)
        ax[row, 4].set_title(f"all-class errors ({100 * (gt != pr).mean():.1f}% px)", fontsize=8)
        for a in ax[row]:
            a.axis("off")
    handles = [Patch(color=palette[c], label=CLASS_NAMES[c]) for c in range(1, num_classes)]
    handles += [Patch(color="red", label=f"{CLASS_NAMES[focus]} false positive"),
                Patch(color="royalblue", label=f"{CLASS_NAMES[focus]} false negative"),
                Patch(color="green", label=f"{CLASS_NAMES[focus]} true positive"),
                Patch(color="gold", label="any misclassified pixel")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))
    plt.savefig(path, dpi=110)
    plt.close(fig)


def plot_dice_by_position(records, info, path, n_bins=10):
    """Pooled prostate Dice per position bin along the gland (0 = base, 1 = apex)."""
    per_patient = defaultdict(lambda: np.zeros((n_bins, 3)))  # inter, pred, gt per bin
    for r in records:
        _, t = position(r, info)
        if t is None:
            continue
        b = min(n_bins - 1, int(t * n_bins))
        per_patient[r["patient"]][b] += (r["inter"], r["pred_p"], r["gt_p"])
    if not per_patient:
        print("[plot] no oriented volumes for dice_by_position, skipped")
        return
    total = sum(per_patient.values())

    def dice(m):
        den = m[:, 1] + m[:, 2]
        return np.where(den > 0, 2 * m[:, 0] / np.maximum(den, 1), np.nan)

    x = (np.arange(n_bins) + 0.5) / n_bins
    plt.figure(figsize=(8, 4.5))
    for p, m in sorted(per_patient.items()):
        plt.plot(x, dice(m), "o-", alpha=0.5, lw=1, label=f"patient {p}")
    plt.plot(x, dice(total), "k-", lw=3, label="all test patients (pooled)")
    plt.axhline(0.60, color="r", ls="--", lw=1, label="A2 end-third floor (0.60)")
    plt.axhline(0.75, color="gray", ls=":", lw=1, label="A1 target (0.75)")
    plt.xlabel("position along prostate (0 = base, nearest bladder -> 1 = apex)")
    plt.ylabel("prostate Dice")
    plt.ylim(0, 1.02)
    plt.title("Where along the gland does segmentation degrade?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_confusion(conf, path, C):
    cm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_NAMES[c] for c in range(C)]
    plt.xticks(range(C), names, rotation=45, ha="right")
    plt.yticks(range(C), names)
    for i in range(C):
        for j in range(C):
            plt.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    plt.xlabel("predicted"); plt.ylabel("ground truth")
    plt.title("Pixel confusion (% of each ground-truth class)")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# --------------------------------------------------------------------------- #
# Single-slice usage example
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_single(model, image_path, mask_path, num_classes, device, out_path):
    raw = load_nifti_2d(image_path)
    h, w = raw.shape
    img = normalise_slice(torch.from_numpy(raw)).unsqueeze(0)
    img, _, _ = pad_to_size(img, torch.zeros(h, w, dtype=torch.long), (256, 144))
    pred = model(img.unsqueeze(0).to(device)).argmax(1)[0, :h, :w].cpu().numpy()
    gt = np.rint(load_nifti_2d(mask_path)).astype(int) if mask_path else None

    palette, alphas = make_palette(num_classes), CLASS_ALPHA[:num_classes]
    vlim = np.percentile(raw, [1, 99.5])
    ncol = 3 if gt is not None else 2
    fig, ax = plt.subplots(1, ncol, figsize=(3.6 * ncol, 5))
    ax[0].imshow(raw, cmap="gray", vmin=vlim[0], vmax=vlim[1]); ax[0].set_title("input")
    show_mask(ax[1], raw, pred, palette, alphas, vlim); ax[1].set_title("prediction")
    if gt is not None:
        show_mask(ax[2], raw, gt, palette, alphas, vlim); ax[2].set_title("ground truth")
    for a in ax:
        a.axis("off")
    fig.legend(handles=[Patch(color=palette[c], label=CLASS_NAMES[c]) for c in range(1, num_classes)],
               loc="lower center", ncol=5, fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    present = {CLASS_NAMES[int(c)]: int((pred == c).sum()) for c in np.unique(pred)}
    print(f"Predicted classes (pixel counts): {present}")
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--root", help="dataset root (needed unless --image is used)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--image", help="predict a single slice instead of a whole split")
    ap.add_argument("--mask", help="optional ground-truth mask for --image")
    ap.add_argument("--out_dir")
    ap.add_argument("--n_cases", type=int, default=6, help="rows per case figure")
    ap.add_argument("--min_area", type=int, default=20, help="min GT prostate pixels for a slice to count as a case")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--prostate_class", type=int, default=PROSTATE_ID)
    ap.add_argument("--rectum_class", type=int, default=RECTUM_ID)
    ap.add_argument("--bladder_class", type=int, default=BLADDER_ID)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    out_dir = Path(args.out_dir or Path(args.run_dir) / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, C, ck = load_model(args.run_dir, args.ckpt, device)
    cls = dict(prostate=args.prostate_class, rectum=args.rectum_class, bladder=args.bladder_class)

    if args.image:
        predict_single(model, args.image, args.mask, C, device, out_dir / "single_prediction.png")
        return
    if not args.root:
        ap.error("--root is required when --image is not given")

    samples = make_splits(args.root)[args.split]
    print(f"Predicting {len(samples)} '{args.split}' slices from "
          f"{len({s['patient'] for s in samples})} patients on {device}")
    d, ms = run_inference(model, samples, C, device, use_amp, args.batch_size)

    # ---- metrics ----
    metrics = SegMetrics(C)
    metrics.update(torch.from_numpy(d["pred"]), torch.from_numpy(d["gt"]), torch.from_numpy(d["valid"]))
    res = metrics.compute()
    records = build_records(d, cls)
    info = volume_info(records)
    regional = regional_analysis(records)
    leak = {}
    pm = (d["pred"] == cls["prostate"]) & d["valid"]
    for name in ("rectum", "bladder"):
        leak[name] = [int((pm & (d["gt"] == cls[name])).sum()), int(pm.sum())]
    crit = check_criteria(res, regional, leak, args, cls)

    print(f"\n=== {args.split.upper()} RESULTS ({args.ckpt}) ===")
    print("class            |  Dice  |  IoU")
    for c in range(C):
        print(f"  {c} {CLASS_NAMES[c]:<12} | {res['dice'][c]:.4f} | {res['iou'][c]:.4f}")
    print(f"mean fg Dice {res['mean_dice_fg']:.4f} | mean IoU {res['mean_iou']:.4f} | "
          f"pixel acc {res['pixel_accuracy']:.4f}")

    print("\nProstate Dice per patient (pooled over all slices/weeks):")
    per_pat = defaultdict(lambda: np.zeros(3))
    for r in records:
        per_pat[r["patient"]] += (r["inter"], r["pred_p"], r["gt_p"])
    pat_dice = {p: float(2 * m[0] / (m[1] + m[2])) if (m[1] + m[2]) > 0 else float("nan")
                for p, m in sorted(per_pat.items())}
    for p, v in pat_dice.items():
        print(f"  patient {p}: {v:.4f}")

    ov = regional["overall"]
    print(f"\nProstate by position along the gland (A2): base {ov['base']:.3f} | mid {ov['mid']:.3f} | "
          f"apex {ov['apex']:.3f}   [{regional['n_volumes']} volumes, "
          f"{regional['unoriented_volumes']} without bladder for orientation]")
    print(f"False-positive prostate on slices outside the GT span: "
          f"{regional['fp_slices_outside_span']}/{regional['slices_outside_span']}")
    print(f"Leakage of predicted prostate into rectum {crit['A3_leakage_rectum']:.2%}, "
          f"bladder {crit['A3_leakage_bladder']:.2%}")
    print("Acceptance criteria:", {k: v for k, v in crit.items() if k.endswith("pass")})
    print(f"Inference: {ms:.2f} ms/slice (batch {args.batch_size}, {'AMP' if use_amp else 'fp32'})")

    # ---- figures ----
    dices = np.array([slice_dice(r) for r in records])
    has_gt = np.array([r["gt_p"] >= args.min_area for r in records])
    cand = np.where(has_gt & ~np.isnan(dices))[0]
    if len(cand):
        order = cand[np.argsort(dices[cand])]
        worst = list(order[:args.n_cases])
        mid = len(order) // 2
        typical = list(order[max(0, mid - args.n_cases // 2): mid + (args.n_cases + 1) // 2])
        draw_cases(typical, d, records, info, cls["prostate"], C, out_dir / "cases_typical.png",
                   "Typical cases: median prostate Dice")
        draw_cases(worst, d, records, info, cls["prostate"], C, out_dir / "cases_worst.png",
                   "Failure autopsy: lowest prostate Dice")
    fp_idx = [k for k, r in enumerate(records) if r["gt_p"] == 0 and r["pred_p"] > 0]
    fp_idx.sort(key=lambda k: -records[k]["pred_p"])
    draw_cases(fp_idx[:args.n_cases], d, records, info, cls["prostate"], C,
               out_dir / "cases_false_positive.png", "Spurious prostate: predicted where none exists")
    plot_dice_by_position(records, info, out_dir / "dice_by_position.png")
    plot_confusion(metrics.conf.numpy(), out_dir / "confusion_matrix.png", C)

    out = dict(split=args.split, ckpt=args.ckpt, epoch=ck["epoch"],
               dice={CLASS_NAMES[c]: nan_to_none([res["dice"][c].item()])[0] for c in range(C)},
               iou={CLASS_NAMES[c]: nan_to_none([res["iou"][c].item()])[0] for c in range(C)},
               prostate_dice_per_patient={str(p): v for p, v in pat_dice.items()},
               regional=regional, leakage=leak, criteria=crit, ms_per_slice=ms,
               n_false_positive_slices=len(fp_idx))
    with open(out_dir / "predict_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved figures and predict_results.json to {out_dir}/")


if __name__ == "__main__":
    main()
