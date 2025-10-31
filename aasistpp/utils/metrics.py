
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = max(fnr[idx], fpr[idx])
    return float(eer), float(thr[idx])

def compute_basic_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    eer, thr = compute_eer(y_true, y_prob)
    return {"acc": acc, "auc": auc, "eer": eer, "thr": thr}
