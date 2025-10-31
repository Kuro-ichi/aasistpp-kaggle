
import torch
import torch.nn as nn
import torch.nn.functional as F

class AASISTPPSmall(nn.Module):
    """
    Một backbone nhỏ lấy cảm hứng AASIST++ cho demo Kaggle.
    Nhận đầu vào: [B, C=3, F, T] (multi-res mel)
    Trả về: logits nhị phân + embedding L2-norm (dim=256 mặc định).
    """
    def __init__(self, in_channels=3, emb_dim=256, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.emb = nn.Linear(256, emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_embedding=False):
        # x: [B, C, F, T]
        h = self.conv(x).view(x.size(0), -1)  # [B, 256]
        emb = self.emb(h)                     # [B, D]
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.cls(emb)                # [B, 2]
        if return_embedding:
            return logits, emb
        return logits
