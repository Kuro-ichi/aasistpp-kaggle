
import os, glob, re
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

_UTT_RE = re.compile(r'(LA_[A-Z]_\d+)')

def _guess_la_root(user_root: Optional[str] = None) -> str:
    cands = []
    if user_root:
        cands.append(user_root)
    cands += [
        "/kaggle/input/asvspoof-2019-dataset/LA/LA",
        "asvspoof-2019-dataset/LA/LA",
        "LA/LA"
    ]
    for c in cands:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "ASVspoof2019_LA_cm_protocols")):
            return os.path.abspath(c)
    raise FileNotFoundError("Không tìm thấy thư mục LA/LA (chứa ASVspoof2019_LA_*).")

@dataclass
class Proto:
    utt_id: str
    label: int  # bonafide=1, spoof=0

def _parse_protocol_line(line: str) -> Optional[Proto]:
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    parts = s.split()
    label = None
    for p in parts[::-1]:
        if p.lower() == 'bonafide':
            label = 1; break
        if p.lower() == 'spoof':
            label = 0; break
    if label is None: 
        return None
    utt = None
    for p in parts:
        if p.startswith('LA_'):
            utt = p; break
    if utt is None:
        for p in parts:
            if p.upper().startswith('LA'):
                utt = p; break
    if utt is None: 
        return None
    return Proto(utt_id=utt, label=label)

def _infer_split_from_protocol(path: str) -> str:
    name = os.path.basename(path).lower()
    if 'train' in name or 'trn' in name:
        return 'ASVspoof2019_LA_train'
    if 'dev' in name or 'trl' in name:
        return 'ASVspoof2019_LA_dev'
    if 'eval' in name or 'tst' in name or 'test' in name:
        return 'ASVspoof2019_LA_eval'
    # fallback
    return 'ASVspoof2019_LA_train'

def load_protocol(protocol_path: str) -> List[Proto]:
    out = []
    with open(protocol_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = _parse_protocol_line(line)
            if p:
                out.append(p)
    if not out:
        raise RuntimeError(f"Không parse được protocol: {protocol_path}")
    return out

def resolve_audio_path(la_root: str, split_dir: str, utt_id: str) -> Optional[str]:
    base = os.path.join(la_root, split_dir)
    for ext in ('.flac', '.wav'):
        for sub in ('flac', 'wav', ''):
            cand = os.path.join(base, sub, utt_id + ext) if sub else os.path.join(base, utt_id + ext)
            if os.path.exists(cand):
                return cand
    # glob fallback
    gl = glob.glob(os.path.join(base, '**', f'{utt_id}.*'), recursive=True)
    return gl[0] if gl else None

class ASVspoofLADataset(Dataset):
    def __init__(self, la_root: str, protocol_path: str, segment_sec: float = 4.0, train: bool = True, sample_rate: int = 16000):
        self.la_root = _guess_la_root(la_root)
        self.protocol_path = protocol_path
        self.segment_sec = segment_sec
        self.train = train
        self.sr = sample_rate
        self.entries: List[Proto] = load_protocol(protocol_path)
        self.split_dir = _infer_split_from_protocol(protocol_path)
        self.target_len = int(self.segment_sec * self.sr)

    def __len__(self):
        return len(self.entries)

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)  # mono
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() >= self.target_len:
            if self.train:
                start = torch.randint(0, wav.numel() - self.target_len + 1, (1,)).item()
            else:
                start = max((wav.numel() - self.target_len)//2, 0)
            return wav[start:start+self.target_len]
        pad = self.target_len - wav.numel()
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(wav, (left, right), mode='constant', value=0.0)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        path = resolve_audio_path(self.la_root, self.split_dir, e.utt_id)
        if path is None:
            raise FileNotFoundError(f"Không thấy audio cho {e.utt_id} trong split {self.split_dir}")
        wav = self._load_wav(path)
        wav = self._crop_or_pad(wav)
        y = torch.tensor(e.label, dtype=torch.long)
        return wav, y, e.utt_id
