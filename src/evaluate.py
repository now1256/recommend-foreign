"""평가 — 전체 행 AUC가 아니라 개인별 랭킹을 주 지표로."""
import numpy as np
from sklearn.metrics import roc_auc_score

def recall_at_k(scores, Y, ks=(1,3,5,10)):
    """scores, Y: (n_user, n_item). 1곳 이상 방문한 사용자만."""
    has = Y.sum(1) > 0
    s, y = scores[has], Y[has]
    order = np.argsort(-s, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(y, order[:, :k], 1).max(1)
        out[f"R@{k}"] = round(float(hit.mean()), 4)
    out["n_users"] = int(has.sum())
    return out

def popularity_baseline(Y_train, Y_test, ks=(1,3,5,10)):
    pop = Y_train.mean(0)
    return recall_at_k(np.tile(pop, (len(Y_test), 1)), Y_test, ks)

def coverage(scores, item_names, k=1):
    """top-k에 등장하는 아이템 종류 수 + 엔트로피 (쏠림 진단)"""
    top = np.argsort(-scores, axis=1)[:, :k].ravel()
    cnt = np.bincount(top, minlength=scores.shape[1]).astype(float)
    p = cnt[cnt > 0] / cnt.sum()
    return {"n_items_topk": int((cnt > 0).sum()), "total_items": scores.shape[1],
            "entropy": round(float(-(p * np.log(p)).sum()), 3),
            "max_entropy": round(float(np.log(scores.shape[1])), 3)}

def auc(scores, Y, w=None):
    return round(float(roc_auc_score(Y.ravel(), scores.ravel(),
                 sample_weight=None if w is None else np.repeat(w, Y.shape[1]))), 4)
