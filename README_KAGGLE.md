
# AASIST++ (Small) — Kaggle-ready for ASVspoof2019 LA

Bộ mã này thiết kế để **chạy mượt trên Kaggle** với cấu trúc data như ảnh:
`asvspoof-2019-dataset/LA/LA/...`.

## Điểm chính
- **Dataset loader** linh hoạt cho ASVspoof2019 LA, tự tìm file audio `.wav/.flac`.
- **Multi-resolution log-mel** (3 kênh) + **backbone AASISTPPSmall** nhẹ để demo.
- Có **train script** lấy **mini subset** cân bằng, tính **loss/ACC/AUC/EER**, và xuất **vector**:
  `*_embeddings.npy`, `*_logits.npy`, `*_probs.npy`, `*_labels.npy`, `*_utts.txt`.
- Cấu hình YAML `configs/kaggle_la.yaml` đã **điền sẵn đường dẫn Kaggle**.

## Cách chạy trên Kaggle
1. Tạo Notebook mới (GPU optional).
2. Add Dataset: `asvspoof-2019-dataset` (đúng cấu trúc `LA/LA/...`).
3. Upload folder này (ít nhất `scripts/train_mini.py` và `configs/kaggle_la.yaml`).
4. Chạy:
   ```bash
   python scripts/train_mini.py --config configs/kaggle_la.yaml --mini_n 80 --epochs 3
   ```

> Bạn có thể đổi `--mini_n` để tăng/giảm số mẫu *mỗi lớp* (bonafide/spoof).

## Kết quả
- `outputs/history.json` — lịch sử loss/acc/auc/eer.
- `outputs/{trainmini,devmini}_*.npy` — embeddings, logits, probs, labels, kèm `*_utts.txt`.
- `outputs/checkpoints/aasistpp_small.pt` — checkpoint PyTorch.

## Gợi ý nâng cấp
- Thay backbone nhỏ bằng mô hình AASIST++ hoàn chỉnh.
- Dùng toàn bộ dữ liệu (bỏ `--mini_n` hoặc set lớn).
- Thêm SpecAugment nâng cao, mixup, scheduler.
