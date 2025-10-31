#!/usr/bin/env python
# coding: utf-8
"""
Mini-train AASIST++ (Kaggle-ready)
- Auto-detect ASVspoof2019-LA path if YAML path invalid
- Filter out missing-audio entries (robust on partial datasets)
- Balanced mini-subset for quick training/debug
- Export vectors + metrics + checkpoint

Usage (Kaggle):
  %cd /kaggle/working/aasistpp-kaggle
  %env PYTHONPATH=/kaggle/working/aasistpp-kaggle:$PYTHONPATH
  !python scripts/train_mini.py --config configs/kaggle_la.yaml --mini_n 80 --epochs 3 --out_dir outputs
"""

import os, argparse, yaml, json, time, glob, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aasistpp.data.asvspoof_la import ASVspoofLADataset, resolve_audio_path
from aasistpp.features.multires_mel import MultiResMelSpec
from aasistpp.models.aasistpp_small import AASISTPPSmall
from aasistpp.utils.spec_augment import spec_augment
from aasistpp.utils.metrics import compute_basic_metrics


# ---------------------- Utils ----------------------
def log(msg: str):
    print(f"[train_mini] {msg}", flush=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def balanced_subset(entries, n_per_class=100, seed=42):
    import random
    random.seed(seed)
    pos = [e for e in entries if e.label == 1]
    neg = [e for e in entries if e.label == 0]
    random.shuffle(pos); random.shuffle(neg)
    if n_per_class <= 0:
        take = pos + neg
    else:
        take = pos[:n_per_class] + neg[:n_per_class]
    random.shuffle(take)
    return take

def filter_existing(ds, entries):
    """Giữ lại những entry có file audio thực sự tồn tại."""
    keep, miss = [], []
    for e in entries:
        p = resolve_audio_path(ds.la_root, ds.split_dir, e.utt_id)
        if p and os.path.exists(p):
            keep.append(e)
        else:
            miss.append(e.utt_id)
    log(f"filter_existing[{ds.split_dir}]: kept={len(keep)} | missing={len(miss)}")
    if miss:
        log("  missing examples (first 10): " + ", ".join(miss[:10]))
    return keep

def shallow_autodetect_la_root():
    """
    Quét nông /kaggle/input để tìm thư mục .../LA/LA có chứa ASVspoof2019_LA_cm_protocols
    (Nhanh, tránh os.walk sâu gây chậm/KeyboardInterrupt)
    """
    cands = []
    cands += glob.glob('/kaggle/input/*/LA/LA')
    cands += glob.glob('/kaggle/input/*/*/LA/LA')
    cands = [c for c in cands if os.path.isdir(os.path.join(c, 'ASVspoof2019_LA_cm_protocols'))]
    return cands[0] if cands else None

def ensure_paths_in_cfg(cfg: dict) -> dict:
    """
    - Nếu cfg['data']['audio_root'] hợp lệ -> dùng luôn
    - Ngược lại -> autodetect rồi gán lại audio_root/protocol_{train,dev}
    """
    data = cfg.get('data', {})
    ar = data.get('audio_root', '')
    use_provided = bool(ar) and os.path.isdir(ar) and os.path.isdir(os.path.join(ar, 'ASVspoof2019_LA_cm_protocols'))

    if use_provided:
        la_root = ar
        proto_dir = os.path.join(la_root, 'ASVspoof2019_LA_cm_protocols')
        train_path = data.get('protocol_train', '')
        dev_path   = data.get('protocol_dev', '')
        # Lấp nếu thiếu
        if not (os.path.isfile(train_path) and os.path.isfile(dev_path)):
            train_cands = [f for f in os.listdir(proto_dir) if f.lower().endswith('.txt') and ('train' in f.lower() or 'trn' in f.lower())]
            dev_cands   = [f for f in os.listdir(proto_dir) if f.lower().endswith('.txt') and ('dev' in f.lower()   or 'trl' in f.lower())]
            assert train_cands and dev_cands, "Không tìm thấy protocol train/dev trong protocols."
            train_path = os.path.join(proto_dir, sorted(train_cands, key=len, reverse=True)[0])
            dev_path   = os.path.join(proto_dir, sorted(dev_cands,   key=len, reverse=True)[0])
        cfg['data']['protocol_train'] = train_path
        cfg['data']['protocol_dev']   = dev_path
        log(f"Using provided audio_root: {la_root}")
        return cfg

    # Autodetect
    la_root = shallow_autodetect_la_root()
    if not la_root:
        raise FileNotFoundError("Không tìm thấy thư mục LA/LA trong /kaggle/input (chứa ASVspoof2019_LA_cm_protocols). Hãy Add dataset đúng.")
    proto_dir = os.path.join(la_root, 'ASVspoof2019_LA_cm_protocols')
    train_cands = [f for f in os.listdir(proto_dir) if f.lower().endswith('.txt') and ('train' in f.lower() or 'trn' in f.lower())]
    dev_cands   = [f for f in os.listdir(proto_dir) if f.lower().endswith('.txt') and ('dev' in f.lower()   or 'trl' in f.lower())]
    assert train_cands and dev_cands, "Không tìm thấy protocol train/dev trong protocols."
    cfg['data']['audio_root']     = la_root
    cfg['data']['protocol_train'] = os.path.join(proto_dir, sorted(train_cands, key=len, reverse=True)[0])
    cfg['data']['protocol_dev']   = os.path.join(proto_dir, sorted(dev_cands,   key=len, reverse=True)[0])
    log(f"Auto-detected audio_root: {la_root}")
    return cfg


# ---------------------- Argparse ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Mini-train AASIST++ (Kaggle-ready)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mini_n", type=int, default=80, help="Số mẫu mỗi lớp từ train/dev để chạy nhanh. 0 = dùng hết.")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs trong config.")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--dev_full", action="store_true", help="Dùng full DEV thay vì mini.")
    return ap.parse_args()


# ---------------------- Main ----------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Đảm bảo path đúng (tin vào audio_root hợp lệ; nếu không thì autodetect)
    cfg = ensure_paths_in_cfg(cfg)

    # Override epochs nếu truyền qua arg
    if args.epochs is not None:
        cfg['train']['epochs'] = int(args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    # --------- Datasets (full) ----------
    tr_ds_full = ASVspoofLADataset(
        la_root=cfg['data']['audio_root'],
        protocol_path=cfg['data']['protocol_train'],
        segment_sec=float(cfg['data'].get('segment_sec', 4.0)),
        train=True,
        sample_rate=int(cfg['sample_rate'])
    )
    de_ds_full = ASVspoofLADataset(
        la_root=cfg['data']['audio_root'],
        protocol_path=cfg['data']['protocol_dev'],
        segment_sec=float(cfg['data'].get('segment_sec', 4.0)),
        train=False,
        sample_rate=int(cfg['sample_rate'])
    )

    # --------- Mini subsets (balanced) + filter missing ----------
    if args.mini_n > 0:
        tr_entries = balanced_subset(tr_ds_full.entries, n_per_class=args.mini_n, seed=42)
        tr_entries = filter_existing(tr_ds_full, tr_entries)
        if not tr_entries:  # Fallback nếu sau lọc không còn mẫu
            tr_entries = filter_existing(tr_ds_full, tr_ds_full.entries)
            log(f"Fallback to all available TRAIN entries: {len(tr_entries)}")

        if args.dev_full:
            de_entries = filter_existing(de_ds_full, de_ds_full.entries)
        else:
            de_entries = balanced_subset(de_ds_full.entries, n_per_class=args.mini_n, seed=43)
            de_entries = filter_existing(de_ds_full, de_entries)
            if not de_entries:
                de_entries = filter_existing(de_ds_full, de_ds_full.entries)
                log(f"Fallback to all available DEV entries: {len(de_entries)}")
    else:
        tr_entries = filter_existing(tr_ds_full, tr_ds_full.entries)
        de_entries = filter_existing(de_ds_full, de_ds_full.entries)

    # --------- Wrap mini/full into Dataset views ----------
    class _Mini(tr_ds_full.__class__):
        def __init__(self, base, entries, train):
            self.__dict__ = base.__dict__.copy()
            self.entries = entries
            self.train = train

    tr_ds = _Mini(tr_ds_full, tr_entries, train=True)
    de_ds = _Mini(de_ds_full, de_entries, train=False)

    # --------- DataLoaders ----------
    num_workers = int(cfg['train'].get('num_workers', 0))
    # Num_workers=0 an toàn cho Kaggle khi debug; bạn có thể tăng sau khi ổn định.
    dl_tr = DataLoader(
        tr_ds, batch_size=int(cfg['train']['batch_size']), shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    dl_de = DataLoader(
        de_ds, batch_size=int(cfg['train']['batch_size']), shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # --------- Feature & Model ----------
    device = args.device
    feat = MultiResMelSpec(
        sample_rate=int(cfg['sample_rate']),
        n_mels=int(cfg['n_mels']),
        n_fft=int(cfg['n_fft']),
        hop_length=int(cfg['hop_length']),
        win_lengths=tuple(cfg['win_lengths'])
    ).to(device)

    model = AASISTPPSmall(
        in_channels=int(cfg['model']['in_channels']),
        emb_dim=int(cfg['model']['emb_dim'])
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['train']['lr']),
        weight_decay=float(cfg['train']['weight_decay'])
    )
    ce = nn.CrossEntropyLoss()

    # ---------------------- Train Loop ----------------------
    history = []
    epochs = int(cfg['train']['epochs'])
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0; nseen = 0
        t0 = time.time()
        for wav, y, _ in dl_tr:
            wav = wav.to(device).float()
            y = y.to(device)
            X = feat(wav)                       # [B, C, F, T]
            X = spec_augment(X, **cfg.get('augment', {}))
            logits = model(X)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item()) * len(y); nseen += len(y)

        # ---------- Eval ----------
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for wav, y, _ in dl_de:
                wav = wav.to(device).float()
                y = y.to(device)
                X = feat(wav)
                logits = model(X)
                prob = torch.softmax(logits, dim=1)[:, 1]
                ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
        ys = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.float32)
        ps = np.concatenate(ps) if ps else np.zeros((0,), dtype=np.float32)

        if len(ys) > 0:
            m = compute_basic_metrics(ys, ps)
        else:
            m = {"acc": 0.0, "auc": 0.5, "eer": 0.5, "thr": 0.5}

        tr_loss = running / max(nseen, 1)
        rec = {"epoch": epoch, "train_loss": tr_loss, **m, "sec": time.time() - t0}
        history.append(rec)
        log(f"[{epoch}/{epochs}] train_loss={tr_loss:.4f} | acc={m['acc']:.3f} | AUC={m['auc']:.3f} | EER={m['eer']*100:.2f}%")

    # ---------------------- Save history ----------------------
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---------------------- Vector Extraction ----------------------
    def extract(split_name, loader):
        all_emb, all_logits, all_probs, all_labels, all_utts = [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for wav, y, utt in loader:
                wav = wav.to(device).float()
                X = feat(wav)
                logits, emb = model(X, return_embedding=True)
                prob = torch.softmax(logits, dim=1)
                all_emb.append(emb.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.numpy())
                all_utts += list(utt)

        if not all_emb:  # không có mẫu
            log(f"[extract:{split_name}] No data, skip.")
            return

        emb = np.concatenate(all_emb, axis=0)
        logits_np = np.concatenate(all_logits, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        np.save(os.path.join(args.out_dir, f"{split_name}_embeddings.npy"), emb)
        np.save(os.path.join(args.out_dir, f"{split_name}_logits.npy"), logits_np)
        np.save(os.path.join(args.out_dir, f"{split_name}_probs.npy"), probs)
        np.save(os.path.join(args.out_dir, f"{split_name}_labels.npy"), labels)
        with open(os.path.join(args.out_dir, f"{split_name}_utts.txt"), "w") as f:
            for u in all_utts:
                f.write(u + "\n")
        log(f"[extract:{split_name}] saved: {len(all_utts)} items")

    extract("trainmini", dl_tr)
    extract("devmini", dl_de)

    # ---------------------- Save checkpoint ----------------------
    ckpt = {"model": model.state_dict(), "cfg": cfg}
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    torch.save(ckpt, os.path.join(args.out_dir, "checkpoints/aasistpp_small.pt"))
    log("✔ Done. Check outputs/ for artifacts.")


if __name__ == "__main__":
    main()
