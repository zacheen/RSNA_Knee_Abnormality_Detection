"""Train KneeNet on the cache from build_cache.py with report-derived labels.

Targets are report labels, hard 0/1 (prompt v1) or soft probabilities (v2), with -1 masked. The 58 radiologist-labeled
studies are the validation set and stay out of training unless --train-on-gt is given (tiny local smoke test only).

Usage
  python src/train.py --cache "data/work/cache/train_*" --labels data/work/report_labels.jsonl --epochs 10
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BAND, CROP_FG, CROP_MM, DATA_ROOT, IMG_SIZE, LABELS, N_SLICES, PLANES, SETTINGS, WORK, code_version  # noqa: E402
from model import KneeNet, WindowNet, mirror_lr, predict_tta, to_input  # noqa: E402
from report_labels import load_labels as load_report_labels  # noqa: E402


def _check_cache(d: Path, meta: pd.DataFrame):
    """Refuse a cache built with other settings than this run's, because the checkpoint records this run's settings.

    crop_fg cannot be seen in the array, so meta.csv is the only evidence. Older caches lack that column and only get the shape check.
    """
    if "crop_fg" in meta and bool(meta.crop_fg.iloc[0]) != CROP_FG:
        raise SystemExit(f"{d} was built with crop_fg={bool(meta.crop_fg.iloc[0])}, this run has {CROP_FG}")
    built = (float(meta.crop_mm.iloc[0]) if "crop_mm" in meta else 0.0,
             [float(x) for x in str(meta.band.iloc[0]).split(",")] if "band" in meta else [0.0, 1.0])
    if built != (CROP_MM, BAND):
        raise SystemExit(f"{d} was built with (crop_mm, band)={built}, this run has {(CROP_MM, BAND)}")
    shape = np.load(d / f"{meta.StudyInstanceUID.iloc[0]}.npy", mmap_mode="r").shape
    if shape != (len(PLANES), N_SLICES, IMG_SIZE, IMG_SIZE):
        raise SystemExit(f"{d} holds {shape}, this run expects {(len(PLANES), N_SLICES, IMG_SIZE, IMG_SIZE)}")


def index_cache(patterns):
    paths, present = {}, {}
    for d in sorted({p for pat in patterns for p in glob.glob(pat)}):
        meta = pd.read_csv(Path(d) / "meta.csv")
        _check_cache(Path(d), meta)
        for r in meta.itertuples(index=False):
            paths[r.StudyInstanceUID] = Path(d) / f"{r.StudyInstanceUID}.npy"
            present[r.StudyInstanceUID] = np.array([getattr(r, f"has_{p}") for p in PLANES], bool)
    return paths, present


class CacheDataset(Dataset):
    def __init__(self, uids, paths, present, targets: pd.DataFrame, augment: bool):
        self.uids, self.paths, self.present, self.targets, self.augment = uids, paths, present, targets, augment

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol = to_input(np.load(self.paths[uid]))
        if self.augment:
            vol = vol.clamp(1e-4, 1).pow(float(np.random.uniform(0.8, 1.25))) * float(np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.5:
                vol = mirror_lr(vol)  # a plain width flip on every plane would be wrong, see mirror_lr
        y = torch.tensor(self.targets.loc[uid].values.astype(np.float32))
        return vol, torch.from_numpy(self.present[uid]), y


def macro_auc(y: np.ndarray, p: np.ndarray):
    aucs = {lab: roc_auc_score(y[:, j], p[:, j]) for j, lab in enumerate(LABELS) if 0 < y[:, j].sum() < len(y)}
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for vol, present, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            ps.append(predict_tta(model, vol.to(device), present.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, help="glob(s) of cache dirs containing meta.csv")
    ap.add_argument("--labels", type=Path, default=WORK / "report_labels.jsonl")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4, help="peak lr for the attention pools and the head")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="peak lr for the pretrained encoder")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--pool", choices=["attn", "conv"], default="attn", help="conv also sees slice order")
    ap.add_argument("--arch", choices=["slices", "windows"], default="slices",
                    help="windows encodes 3-slice RGB windows with per-finding attention, see WindowNet")
    ap.add_argument("--crop", type=int, default=288, help="windows only, square crop taken from the cached slices")
    ap.add_argument("--train-windows", type=int, default=32, help="windows only, windows encoded per study per step")
    ap.add_argument("--avg-last", type=int, default=3, help="keep this many final epochs and average their predictions")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-on-gt", action="store_true")
    ap.add_argument("--out", type=Path, default=WORK / "models" / "kneenet.pt")
    ap.add_argument("--budget-hours", type=float, default=11.0, help="wall-clock cap for the training loop")
    ap.add_argument("--seed", type=int, default=0, help="init, shuffle order and augmentation draws")
    args = ap.parse_args()
    if args.avg_last < 1:
        ap.error("--avg-last must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, present = index_cache(args.cache)
    rep = load_report_labels(args.labels)
    gt = pd.read_csv(DATA_ROOT / "train.csv").set_index("StudyInstanceUID")
    gt = gt[gt[LABELS].notna().all(axis=1)][LABELS].astype(np.float32)

    gt_ids = set(gt.index)
    train_ids = [u for u in paths if u in rep.index and (args.train_on_gt or u not in gt_ids)]
    val_ids = [u for u in paths if u in gt_ids]
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={gpu} cached={len(paths)} train={len(train_ids)} val(gt)={len(val_ids)}", flush=True)
    if not train_ids:
        raise SystemExit("no training studies: need cached studies that also have report labels")

    dl_tr = DataLoader(CacheDataset(train_ids, paths, present, rep.astype(np.float32), True),
                       batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=len(train_ids) > args.bs)
    dl_va = DataLoader(CacheDataset(val_ids, paths, present, gt, False), batch_size=args.bs,
                       num_workers=args.workers) if val_ids else None

    window_args = {"crop": args.crop, "train_windows": args.train_windows}
    if args.arch == "windows":
        model = WindowNet(args.backbone, pretrained=True, n_slots=len(PLANES), **window_args).to(device)
    else:
        model = KneeNet(args.backbone, pretrained=True, dropout=args.dropout, pool=args.pool, n_slots=len(PLANES)).to(device)
    enc = list(model.encoder.parameters())
    enc_ids = {id(t) for t in enc}
    rest = [t for t in model.parameters() if id(t) not in enc_ids]
    groups = [{"params": enc, "lr": args.backbone_lr}, {"params": rest, "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr],
                                                total_steps=args.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Average the last avg_last epochs instead of picking the best one. On 58 validation studies neighbouring epochs
    # differ by about as much as the gains being chased, so the "best" epoch would mostly be noise.
    kept = []  # (checkpoint path, validation probabilities), oldest first
    t_start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Epoch files left by an earlier run would be globbed and averaged in by the submit kernel, silently mixing two models.
    for stale in [args.out, *args.out.parent.glob(f"{args.out.stem}_ep*{args.out.suffix}")]:
        stale.unlink(missing_ok=True)
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for vol, pres, y in dl_tr:
            vol, pres, y = vol.to(device), pres.to(device), y.to(device)
            mask = (y >= 0).float()  # -1 is "report does not address it", so it carries no gradient
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(vol, pres)
            loss = (bce(logits.float(), y.clamp(min=0)) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"epoch {ep + 1}/{args.epochs} loss={tot / max(n, 1):.4f} {time.time() - t0:.0f}s"
        y_va = p_va = None
        if dl_va:
            y_va, p_va = predict(model, dl_va, device)
            msg += f" val_macro_auc={macro_auc(y_va, p_va)[0]:.4f}"
        print(msg, flush=True)
        ck = args.out.with_name(f"{args.out.stem}_ep{ep + 1:02d}{args.out.suffix}")
        torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "pool": args.pool, "arch": args.arch,
                    "window_args": window_args, "code_version": code_version(), **SETTINGS}, ck)
        kept.append((ck, p_va))
        if len(kept) > args.avg_last:
            kept.pop(0)[0].unlink()  # the submit kernel averages every kneenet*.pt it finds, so evicted epochs must not linger
        # A kernel killed at Kaggle's 12 h cap saves no output at all, so stop while the next epoch still fits.
        epoch_s = time.time() - t0
        if time.time() - t_start + epoch_s > args.budget_hours * 3600 and ep + 1 < args.epochs:
            print(f"[budget] stopping after epoch {ep + 1}, the next one would pass {args.budget_hours} h", flush=True)
            break
    if dl_va:
        score, per = macro_auc(y_va, np.mean([pv for _, pv in kept], axis=0))
        print(f"averaged last {len(kept)} epochs: val_macro_auc={score:.4f}", {k: round(v, 3) for k, v in per.items()})
    print("saved", [str(c) for c, _ in kept])


if __name__ == "__main__":
    main()
