
#!/usr/bin/env python
# coding: utf-8
import os, argparse, yaml, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from aasistpp.data.asvspoof_la import ASVspoofLADataset
from aasistpp.features.multires_mel import MultiResMelSpec
from aasistpp.models.aasistpp_small import AASISTPPSmall
from aasistpp.utils.spec_augment import spec_augment
from aasistpp.utils.metrics import compute_basic_metrics

def set_seed(seed:int=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def balanced_subset(entries, n_per_class=100, seed=42):
    import random
    random.seed(seed)
    pos = [e for e in entries if e.label==1]
    neg = [e for e in entries if e.label==0]
    random.shuffle(pos); random.shuffle(neg)
    take = pos[:n_per_class] + neg[:n_per_class]
    random.shuffle(take)
    return take

def parse_args():
    ap = argparse.ArgumentParser("Mini-train AASIST++ (Kaggle-ready)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mini_n", type=int, default=80, help="Số mẫu mỗi lớp từ train/dev để chạy nhanh.")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs trong config.")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg['train']['epochs'] = int(args.epochs)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    # Datasets
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
    # Mini subsets
    tr_entries = balanced_subset(tr_ds_full.entries, n_per_class=args.mini_n, seed=42)
    de_entries = balanced_subset(de_ds_full.entries, n_per_class=args.mini_n, seed=43)

    # Wrap mini entries into lightweight datasets by monkey-patching __len__/__getitem__.
    class _Mini(tr_ds_full.__class__):
        def __init__(self, base, entries, train):
            self.__dict__ = base.__dict__.copy()
            self.entries = entries
            self.train = train
    tr_ds = _Mini(tr_ds_full, tr_entries, train=True)
    de_ds = _Mini(de_ds_full, de_entries, train=False)

    dl_tr = DataLoader(tr_ds, batch_size=int(cfg['train']['batch_size']), shuffle=True,
                       num_workers=int(cfg['train']['num_workers']), pin_memory=True)
    dl_de = DataLoader(de_ds, batch_size=int(cfg['train']['batch_size']), shuffle=False,
                       num_workers=int(cfg['train']['num_workers']), pin_memory=True)

    # Feature extractor
    feat = MultiResMelSpec(
        sample_rate=int(cfg['sample_rate']),
        n_mels=int(cfg['n_mels']),
        n_fft=int(cfg['n_fft']),
        hop_length=int(cfg['hop_length']),
        win_lengths=tuple(cfg['win_lengths'])
    ).to(args.device)

    # Model
    model = AASISTPPSmall(
        in_channels=int(cfg['model']['in_channels']),
        emb_dim=int(cfg['model']['emb_dim'])
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
    ce = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, int(cfg['train']['epochs'])+1):
        model.train()
        running = 0.0; n = 0
        t0 = time.time()
        for wav, y, _ in dl_tr:
            wav = wav.to(args.device).float()
            y = y.to(args.device)
            X = feat(wav)               # [B, C, F, T]
            X = spec_augment(X, **cfg.get('augment', {}))
            logits = model(X)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item())*len(y); n += len(y)

        # Eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for wav, y, _ in dl_de:
                wav = wav.to(args.device).float()
                y = y.to(args.device)
                X = feat(wav)
                logits = model(X)
                prob = torch.softmax(logits, dim=1)[:,1]
                ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = compute_basic_metrics(ys, ps)
        tr_loss = running/max(n,1)
        history.append({"epoch": epoch, "train_loss": tr_loss, **m, "sec": time.time()-t0})
        print(f"[{epoch}/{cfg['train']['epochs']}] train_loss={tr_loss:.4f} | acc={m['acc']:.3f} | AUC={m['auc']:.3f} | EER={m['eer']*100:.2f}%")

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Extract vectors (train mini + dev mini)
    def extract(split_name, loader):
        all_emb, all_logits, all_probs, all_labels, all_utts = [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for wav, y, utt in loader:
                wav = wav.to(args.device).float()
                X = feat(wav)
                logits, emb = model(X, return_embedding=True)
                prob = torch.softmax(logits, dim=1)
                all_emb.append(emb.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.numpy())
                all_utts += list(utt)
        import numpy as np
        emb = np.concatenate(all_emb, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        np.save(os.path.join(args.out_dir, f"{split_name}_embeddings.npy"), emb)
        np.save(os.path.join(args.out_dir, f"{split_name}_logits.npy"), logits)
        np.save(os.path.join(args.out_dir, f"{split_name}_probs.npy"), probs)
        np.save(os.path.join(args.out_dir, f"{split_name}_labels.npy"), labels)
        with open(os.path.join(args.out_dir, f"{split_name}_utts.txt"), "w") as f:
            for u in all_utts: f.write(u+"\n")

    extract("trainmini", dl_tr)
    extract("devmini", dl_de)

    # Save model checkpoint
    ckpt = {"model": model.state_dict(), "cfg": cfg}
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    torch.save(ckpt, os.path.join(args.out_dir, "checkpoints/aasistpp_small.pt"))

if __name__ == "__main__":
    main()
