
import torch
import torchaudio

class MultiResMelSpec(torch.nn.Module):
    """
    Multi-resolution log-mel spectrogram.
    Given a mono waveform Tensor [B, T] (float32, 16k), returns [B, C, n_mels, Tm].
    C equals len(win_lengths).
    """
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160,
                 win_lengths=(400, 640, 800), fmin=20.0, fmax=None):
        super().__init__()
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax if fmax is not None else sample_rate//2
        self.specs = torch.nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=w,
                hop_length=hop_length,
                f_min=fmin, f_max=self.fmax,
                n_mels=n_mels,
                power=2.0,
                center=True,
                norm="slaney",
                mel_scale="htk"
            ) for w in win_lengths
        ])
        self.amin = 1e-10

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T]
        outs = []
        for spec in self.specs:
            S = spec(wav)                              # [B, n_mels, Tm]
            S = torch.clamp(S, min=self.amin).log()    # natural log
            outs.append(S)
        out = torch.stack(outs, dim=1)                 # [B, C, n_mels, Tm]
        return out
