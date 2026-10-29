"""
train.py -- train, validate, test and save the 2D U-Net baseline on HipMRI slices.

Features
--------
* Patient-level splits and loaders from dataset.py; model, loss and metrics from modules.py.
* AMP on GPU, AdamW + cosine LR, gradient clipping.
* Model selection by validation mean foreground Dice -> ``best.pt``.
* Plots losses and metrics every epoch (training_curves.png), history in JSON/CSV.
* Final test evaluation: per-class Dice/IoU, base/mid/apex prostate Dice (criterion A2),
  boundary leakage into rectum/bladder (A3), VRAM, training time, inference time (A4).
"""

import argparse
import csv
import json
import random
import signal
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import (BLADDER_ID, CLASS_NAMES, DEFAULT_NUM_CLASSES, PROSTATE_ID, RECTUM_ID,
                     get_dataloaders)
from modules import DiceCELoss, SegMetrics, UNet, count_parameters, leakage_counts

STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[train] SIGTERM received: will stop after the current epoch.", flush=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, loss_fn, opt, scaler, device, use_amp, clip=1.0):
    model.train()
    total, n = 0.0, 0
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        y = b["mask"].to(device, non_blocking=True)
        v = b["valid"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y, v)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes, use_amp, cls_ids=None):
    """Evaluate on a loader. If ``cls_ids`` (dict prostate/bladder/rectum) is given, also
    collect per-slice prostate statistics and leakage counts for the test analysis."""
    model.eval()
    metrics = SegMetrics(num_classes)
    total_loss, n_batches = 0.0, 0
    records = []
    leak = {"rectum": [0, 0], "bladder": [0, 0]}
    infer_time, infer_slices = 0.0, 0
    for bi, b in enumerate(loader):
        x = b["image"].to(device, non_blocking=True)
        y = b["mask"].to(device, non_blocking=True)
        v = b["valid"].to(device, non_blocking=True)
        sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
        sync(device)
        if bi > 0:  # skip first batch (warm-up)
            infer_time += time.perf_counter() - t0
            infer_slices += x.shape[0]
        logits = logits.float()
        total_loss += loss_fn(logits, y, v).item()
        n_batches += 1
        pred = logits.argmax(1)
        metrics.update(pred.cpu(), y.cpu(), v.cpu())

        if cls_ids is not None:
            pc, bl, rc = cls_ids["prostate"], cls_ids["bladder"], cls_ids["rectum"]
            pm, gm = (pred == pc) & v, (y == pc) & v
            records += [dict(patient=a, week=w, slice=s, inter=i, pred_p=p, gt_p=t, gt_bladder=g)
                        for a, w, s, i, p, t, g in zip(
                            b["patient"].tolist(), b["week"].tolist(), b["slice"].tolist(),
                            (pm & gm).sum((1, 2)).tolist(), pm.sum((1, 2)).tolist(),
                            gm.sum((1, 2)).tolist(), ((y == bl) & v).sum((1, 2)).tolist())]
            for name, c in (("rectum", rc), ("bladder", bl)):
                num, den = leakage_counts(pred, y, v, pc, [c])
                leak[name][0] += num
                leak[name][1] += den
    out = metrics.compute()
    out["loss"] = total_loss / max(n_batches, 1)
    out["ms_per_slice_batched"] = 1e3 * infer_time / infer_slices if infer_slices else None
    if cls_ids is not None:
        out["records"], out["leak"] = records, leak
    return out


# --------------------------------------------------------------------------- #
# A2: base / mid / apex analysis
# --------------------------------------------------------------------------- #
def _dice(inter, p, t):
    return 2 * inter / (p + t) if (p + t) > 0 else float("nan")


def regional_analysis(records, min_slices=3):
    """For each patient-week volume, split the span of slices containing GT prostate into
    thirds and compute pooled Dice per third. The base is the third nearest the bladder
    (determined from GT bladder location). Also counts false-positive prostate slices
    outside the GT span."""
    vols = defaultdict(list)
    for r in records:
        vols[(r["patient"], r["week"])].append(r)

    per_vol, skipped, unoriented = [], 0, 0
    fp_slices, outside_slices = 0, 0
    for (pat, week), rs in vols.items():
        gt_idx = [r["slice"] for r in rs if r["gt_p"] > 0]
        if len(gt_idx) < min_slices:
            skipped += 1
            continue
        lo, hi = min(gt_idx), max(gt_idx)
        n = hi - lo + 1
        for r in rs:
            if r["slice"] < lo or r["slice"] > hi:
                outside_slices += 1
                fp_slices += int(r["pred_p"] > 0)
        bl_w = sum(r["gt_bladder"] for r in rs)
        if bl_w == 0:
            unoriented += 1
            continue
        bl_mean = sum(r["slice"] * r["gt_bladder"] for r in rs) / bl_w
        high_is_base = bl_mean > (lo + hi) / 2
        acc = [[0, 0, 0] for _ in range(3)]  # per third (by increasing slice idx): inter, pred, gt
        for r in rs:
            if lo <= r["slice"] <= hi:
                k = min(2, int(3 * (r["slice"] - lo) / n))
                acc[k][0] += r["inter"]; acc[k][1] += r["pred_p"]; acc[k][2] += r["gt_p"]
        names = ["apex", "mid", "base"] if high_is_base else ["base", "mid", "apex"]
        per_vol.append(dict(patient=pat, week=week, span=(lo, hi),
                            **{names[k]: _dice(*acc[k]) for k in range(3)}))

    # average volumes -> per patient -> across patients
    by_pat = defaultdict(list)
    for v in per_vol:
        by_pat[v["patient"]].append(v)

    def nanmean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    patients = {p: {k: nanmean([v[k] for v in vs]) for k in ("base", "mid", "apex")}
                for p, vs in by_pat.items()}
    overall = {k: nanmean([d[k] for d in patients.values()]) for k in ("base", "mid", "apex")}
    return dict(overall=overall, per_patient=patients, n_volumes=len(per_vol),
                skipped_volumes=skipped, unoriented_volumes=unoriented,
                fp_slices_outside_span=fp_slices, slices_outside_span=outside_slices)


def check_criteria(test_res, regional, leak, args, cls_ids):
    """Evaluate acceptance criteria A1-A3 against thresholds from the feasibility review."""
    dice = test_res["dice"].tolist()
    ov = regional["overall"]
    worst_end = min(ov["base"], ov["apex"])
    leak_all = (leak["rectum"][0] + leak["bladder"][0]) / max(leak["rectum"][1], 1)
    return {
        "A1_prostate_dice": dice[cls_ids["prostate"]],
        "A1_pass": dice[cls_ids["prostate"]] >= 0.75,
        "A1_all_classes_pass": all(d >= 0.75 for d in dice if d == d),
        "A2_base": ov["base"], "A2_mid": ov["mid"], "A2_apex": ov["apex"],
        "A2_pass": bool(ov["base"] >= 0.60 and ov["apex"] >= 0.60 and (ov["mid"] - worst_end) <= 0.15),
        "A3_leakage_fraction": leak_all,
        "A3_leakage_rectum": leak["rectum"][0] / max(leak["rectum"][1], 1),
        "A3_leakage_bladder": leak["bladder"][0] / max(leak["bladder"][1], 1),
        "A3_pass": leak_all <= 0.05,
    }


# --------------------------------------------------------------------------- #
# Logging / plotting / checkpoints
# --------------------------------------------------------------------------- #
def save_history(history, out_dir):
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    if history:
        keys = [k for k in history[0] if not isinstance(history[0][k], list)]
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for h in history:
                w.writerow({k: h[k] for k in keys})


def plot_history(history, out_dir, num_classes):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(title="Loss (Dice+CE)", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_mean_fg_dice"] for h in history], label="mean fg Dice")
    ax[1].plot(ep, [h["val_mean_iou"] for h in history], label="mean IoU")
    ax[1].set(title="Validation metrics", xlabel="epoch", ylim=(0, 1)); ax[1].legend()
    for c in range(num_classes):
        ax[2].plot(ep, [h["val_dice_per_class"][c] for h in history], label=f"class {c}")
    ax[2].set(title="Validation Dice per class", xlabel="epoch", ylim=(0, 1)); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def save_checkpoint(path, model, opt, sched, scaler, epoch, history, best, bad_epochs):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, history=history,
                    best=best, bad_epochs=bad_epochs), path)


def nan_to_none(xs):
    return [None if x != x else x for x in xs]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="runs/unet")
    ap.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15, help="early-stop after N epochs without val improvement (0 = off)")
    ap.add_argument("--time_limit_h", type=float, default=0.0, help="stop gracefully after this many hours (0 = off)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--resplit", action="store_true", help="ignore provided folders; patient-level re-split")
    # Class ids confirmed by visual label check (see dataset.CLASS_NAMES)
    ap.add_argument("--prostate_class", type=int, default=PROSTATE_ID)
    ap.add_argument("--rectum_class", type=int, default=RECTUM_ID)
    ap.add_argument("--bladder_class", type=int, default=BLADDER_ID)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    set_seed(args.seed)
    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  AMP: {use_amp}")
    cls_ids = dict(prostate=args.prostate_class, rectum=args.rectum_class, bladder=args.bladder_class)
    print(f"Class ids: {cls_ids}")

    loaders, splits = get_dataloaders(args.root, batch_size=args.batch_size, num_classes=args.num_classes,
                                      use_provided=not args.resplit, seed=args.seed,
                                      num_workers=args.num_workers, cache=not args.no_cache)
    print({k: f"{len(v)} slices / {len({s['patient'] for s in v})} patients" for k, v in splits.items()})

    model = UNet(1, args.num_classes, base=args.base, dropout=args.dropout).to(device)
    print(f"U-Net base={args.base}: {count_parameters(model) / 1e6:.2f} M parameters")
    loss_fn = DiceCELoss(args.num_classes, include_background_in_dice=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, bad_epochs, start_epoch = [], -1.0, 0, 1
    if args.resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, best, bad_epochs, start_epoch = ck["history"], ck["best"], ck["bad_epochs"], ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']} (best val fg Dice {best:.4f})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------ training ------------------------------ #
    if not args.test_only:
        stopped_early = None
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, loaders["train"], loss_fn, opt, scaler, device, use_amp)
            val = evaluate(model, loaders["val"], loss_fn, device, args.num_classes, use_amp)
            sched.step()
            history.append(dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tr_loss,
                                val_loss=val["loss"], val_mean_fg_dice=val["mean_dice_fg"],
                                val_mean_iou=val["mean_iou"],
                                val_dice_per_class=nan_to_none(val["dice"].tolist()),
                                val_iou_per_class=nan_to_none(val["iou"].tolist()),
                                epoch_time_s=time.time() - t0))
            improved = val["mean_dice_fg"] > best
            if improved:
                best, bad_epochs = val["mean_dice_fg"], 0
                torch.save(dict(model=model.state_dict(), epoch=epoch, val_fg_dice=best, args=vars(args)),
                           out_dir / "best.pt")
            else:
                bad_epochs += 1
            save_checkpoint(out_dir / "last.pt", model, opt, sched, scaler, epoch, history, best, bad_epochs)
            save_history(history, out_dir)
            plot_history(history, out_dir, args.num_classes)
            dpc = " ".join("-" if d is None else f"{d:.2f}" for d in history[-1]["val_dice_per_class"])
            print(f"[{epoch:3d}/{args.epochs}] train {tr_loss:.4f} | val {val['loss']:.4f} | "
                  f"fg Dice {val['mean_dice_fg']:.4f}{' *' if improved else ''} | per-class {dpc} | "
                  f"{time.time() - t0:.0f}s", flush=True)

            if args.patience and bad_epochs >= args.patience:
                stopped_early = f"early stop (no improvement for {bad_epochs} epochs)"; break
            if STOP_REQUESTED:
                stopped_early = "SIGTERM"; break
            if args.time_limit_h and (time.time() - t_start) / 3600 > args.time_limit_h:
                stopped_early = f"time limit {args.time_limit_h} h"; break
        if stopped_early and epoch < args.epochs and not stopped_early.startswith("early"):
            print(f"[train] Stopped: {stopped_early}. Resubmit with --resume to continue.")
            return
        print(f"Training finished ({stopped_early or 'all epochs'}).")

    train_time_h = (time.time() - t_start) / 3600
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    # -------------------------------- test -------------------------------- #
    if not (out_dir / "best.pt").exists():
        print("No best.pt found; nothing to test."); return
    ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    print(f"Testing best checkpoint (epoch {ck['epoch']}, val fg Dice {ck['val_fg_dice']:.4f})")
    test = evaluate(model, loaders["test"], loss_fn, device, args.num_classes, use_amp, cls_ids)
    regional = regional_analysis(test["records"])
    crit = check_criteria(test, regional, test["leak"], args, cls_ids)

    # batch-1 inference latency on a few test slices
    model.eval()
    n_lat = min(60, len(loaders["test"].dataset))
    xs = [loaders["test"].dataset[i]["image"].unsqueeze(0).to(device) for i in range(n_lat)]
    with torch.no_grad():
        for x in xs[:10]:
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for x in xs[10:]:
            model(x)
        sync(device)
    lat = 1e3 * (time.perf_counter() - t0) / max(len(xs) - 10, 1)

    print("\n=== TEST RESULTS (best checkpoint) ===")
    print("class            | Dice   | IoU")
    for c in range(args.num_classes):
        d, i = test["dice"][c].item(), test["iou"][c].item()
        print(f"  {c} {CLASS_NAMES.get(c, '?'):<12} | {d:.4f} | {i:.4f}")
    print(f"mean fg Dice {test['mean_dice_fg']:.4f} | mean IoU {test['mean_iou']:.4f}")
    print(f"Prostate (class {args.prostate_class}) regional Dice: base {regional['overall']['base']:.3f} | "
          f"mid {regional['overall']['mid']:.3f} | apex {regional['overall']['apex']:.3f}")
    print(f"Leakage: rectum {crit['A3_leakage_rectum']:.3%}, bladder {crit['A3_leakage_bladder']:.3%}")
    print("Criteria:", {k: v for k, v in crit.items() if k.endswith("pass")})
    vr = f"{peak_vram_gb:.2f} GB" if peak_vram_gb is not None else "n/a"
    print(f"Peak VRAM (this process): {vr} | wall time: {train_time_h:.2f} h | "
          f"inference: {lat:.1f} ms/slice (batch 1), {test['ms_per_slice_batched']} ms/slice (batched)")

    results = dict(best_epoch=ck["epoch"], best_val_fg_dice=ck["val_fg_dice"],
                   test_dice=nan_to_none(test["dice"].tolist()), test_iou=nan_to_none(test["iou"].tolist()),
                   test_mean_fg_dice=test["mean_dice_fg"], test_mean_iou=test["mean_iou"],
                   regional=regional, leakage=test["leak"], criteria=crit,
                   assumed_class_ids=cls_ids, params_m=count_parameters(model) / 1e6,
                   peak_vram_gb=peak_vram_gb, wall_time_h=train_time_h,
                   inference_ms_batch1=lat, inference_ms_batched=test["ms_per_slice_batched"],
                   amp=use_amp, base=args.base, batch_size=args.batch_size)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
