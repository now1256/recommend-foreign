"""EASE POI 추천 재현 실험.

실행:
    python -m src.ease_experiment

수도권 외 관광형 방문만 사용하고, 10명 이상이 방문한 POI를 후보로 제한한다.
후보 POI를 2개 이상 방문한 사용자마다 한 곳을 test로 숨긴 leave-one-out 평가다.
나머지 방문은 train에 남기며, 한 곳만 방문한 사용자는 train에만 사용한다.
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C, data as D


def hit_rate(scores, truth, ks=(1, 5, 10, 20)):
    order = np.argsort(-scores, axis=1)
    return {
        f"HR@{k}": round(float(np.mean([t in row[:k] for t, row in zip(truth, order)])), 4)
        for k in ks
    }


def run(min_visitors=10, reg=500.0, seed=0):
    visits = D.aihub_visits().copy()
    visits = visits.dropna(subset=["TRAVELER_ID", "VISIT_AREA_NM", "sgg"])
    visits["TRAVELER_ID"] = visits["TRAVELER_ID"].astype(str)
    visits["sido"] = visits["sgg"].str.split().str[0]
    visits = visits[~visits["sido"].isin(C.CAPITAL)]

    # 기존 실험과 동일하게 POI 이름을 아이템 키로 사용한다.
    pairs = visits[["TRAVELER_ID", "VISIT_AREA_NM"]].drop_duplicates()
    item_n = pairs.groupby("VISIT_AREA_NM")["TRAVELER_ID"].nunique()
    keep_items = item_n[item_n >= min_visitors].index
    pairs = pairs[pairs["VISIT_AREA_NM"].isin(keep_items)].copy()

    users = sorted(pairs["TRAVELER_ID"].unique())
    items = sorted(pairs["VISIT_AREA_NM"].unique())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {x: i for i, x in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=np.float64)
    for u, x in pairs.itertuples(index=False):
        R[user_index[u], item_index[x]] = 1.0

    counts = R.sum(axis=1)
    eval_users = np.flatnonzero(counts >= 2)
    train = R.copy()
    truth = np.empty(len(eval_users), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for n, u in enumerate(eval_users):
        seen = np.flatnonzero(R[u])
        truth[n] = rng.choice(seen)
        train[u, truth[n]] = 0.0

    t0 = time.time()
    gram = train.T @ train + reg * np.eye(train.shape[1])
    precision = np.linalg.inv(gram)
    B = precision / (-np.diag(precision)[None, :])
    np.fill_diagonal(B, 0.0)
    ease_scores = train[eval_users] @ B
    ease_scores[train[eval_users] > 0] = -np.inf
    ease_sec = time.time() - t0

    popularity = train.sum(axis=0)
    pop_scores = np.tile(popularity, (len(eval_users), 1)).astype(float)
    pop_scores[train[eval_users] > 0] = -np.inf

    # 모델 1에서 시도가 이미 정해졌다고 가정한 조건부 평가.
    # 동명 POI가 여러 시도에 있으면 전체 방문에서 가장 빈번한 시도를 대표값으로 쓴다.
    item_sido = (
        visits[visits["VISIT_AREA_NM"].isin(items)]
        .groupby("VISIT_AREA_NM")["sido"]
        .agg(lambda x: x.mode().iloc[0])
        .reindex(items)
        .to_numpy()
    )
    ease_sido_scores = ease_scores.copy()
    pop_sido_scores = pop_scores.copy()
    for n, target in enumerate(truth):
        outside = item_sido != item_sido[target]
        ease_sido_scores[n, outside] = -np.inf
        pop_sido_scores[n, outside] = -np.inf

    result = {
        "data": {
            "raw_tourism_visits": int(len(D.aihub_visits())),
            "users": len(users),
            "items": len(items),
            "interactions_before_holdout": int(R.sum()),
            "train_interactions": int(train.sum()),
            "test_interactions": len(eval_users),
            "eval_users": len(eval_users),
            "min_visitors": min_visitors,
            "excluded_regions": C.CAPITAL,
        },
        "split": "leave-one-out per user with >=2 eligible POIs; seed=0",
        "ease": {"lambda": reg, "fit_sec": round(ease_sec, 4), **hit_rate(ease_scores, truth)},
        "popularity": hit_rate(pop_scores, truth),
        "within_true_sido": {
            "description": "모델 1이 정답 시도를 선택했다고 가정하고 같은 시도 POI만 후보로 평가",
            "ease": hit_rate(ease_sido_scores, truth),
            "popularity": hit_rate(pop_sido_scores, truth),
        },
    }
    C.CACHE.mkdir(parents=True, exist_ok=True)
    (C.CACHE / "ease_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
