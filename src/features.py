"""학습표 구성 — 내국인(사전학습용) / 외국인(파인튜닝용) 두 도메인."""
import numpy as np, pandas as pd
from . import config as C, data as D

# ══ 아이템: 시군구 피처 (두 도메인 공유) ═════════════════════════
def item_matrix(sggs, standardize=True) -> pd.DataFrame:
    """시군구 → 카드 + AI Hub 성격 + 입장객 + 관광자원"""
    V = D.aihub_visits()
    ai = pd.concat([
        pd.crosstab(V.sgg, V.유형, normalize="index").reindex(columns=list(C.VISIT_TYPE.values()), fill_value=0),
        V.groupby("sgg")[["DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION"]].mean(),
        V.groupby("sgg").agg(stay=("RESIDENCE_TIME_MIN", "mean"), npoi=("VISIT_AREA_NM", "nunique")),
        pd.crosstab(V.sgg, V.VISIT_CHC_REASON_CD, normalize="index").add_prefix("why"),
    ], axis=1)
    try: card = D.card_profiles("gu")
    except Exception: card = pd.DataFrame()
    parts = [ai] + [x for x in (card, D.entrance_stats(), D.resource_stats()) if len(x)]

    def align_level(p):
        """시군구 요청은 그대로, 시도 요청은 해당 시군구의 평균 피처로 정렬."""
        rows = []
        for name in sggs:
            if name in p.index:
                rows.append(p.loc[name])
            elif " " not in name:
                mask = p.index.to_series().astype(str).str.split().str[0].eq(name).values
                rows.append(p.loc[mask].mean() if mask.any() else pd.Series(0.0, index=p.columns))
            else:
                rows.append(pd.Series(0.0, index=p.columns))
        return pd.DataFrame(rows, index=sggs, columns=p.columns)

    M = pd.concat([align_level(p) for p in parts], axis=1).fillna(0)
    M = M.loc[:, M.std() > 1e-9]
    if standardize: M = ((M - M.mean()) / (M.std() + 1e-9)).fillna(0)
    return M

# ══ 도메인 A: 내국인 (사전학습) ══════════════════════════════════
KR_CAT = ["gender", "age", "style1", "motive"]
KR_NUM = [f"TRAVEL_STYL_{i}" for i in range(2, 9)] + ["companions"]

def korean_table():
    """반환: 여행객×시군구 롱포맷 + 방문라벨 + 만족도라벨"""
    V, M = D.aihub_visits().copy(), D.aihub_master().copy()
    # 권역별 원본 CSV에서 ID가 숫자/문자열로 혼재하므로 통합 전에 문자열로 고정한다.
    V = V.dropna(subset=["TRAVELER_ID"])
    V["TRAVELER_ID"] = V["TRAVELER_ID"].astype(str)
    M = M.dropna(subset=["TRAVELER_ID"])
    M["TRAVELER_ID"] = M["TRAVELER_ID"].astype(str)
    users = sorted(V.TRAVELER_ID.unique()); sggs = sorted(V.sgg.unique())
    ui = {u: i for i, u in enumerate(users)}; gi = {g: i for i, g in enumerate(sggs)}
    Y = np.zeros((len(users), len(sggs)), "float32")
    S = np.full((len(users), len(sggs)), np.nan, "float32")
    for (u, g), s in V.groupby(["TRAVELER_ID", "sgg"]).DGSTFN.mean().items():
        Y[ui[u], gi[g]] = 1; S[ui[u], gi[g]] = s
    m = M.set_index("TRAVELER_ID").reindex(users)
    U = pd.DataFrame(index=users)
    U["gender"] = (m.GENDER == "여").astype(int).values
    U["age"] = (m.AGE_GRP / 10 - 2).clip(0, 4).fillna(2).astype(int).values
    U["style1"] = (m.TRAVEL_STYL_1 - 1).clip(0, 6).fillna(3).astype(int).values
    U["motive"] = (m.TRAVEL_MOTIVE_1.fillna(10) - 1).clip(0, 9).astype(int).values
    num = pd.DataFrame(index=users)
    for c in KR_NUM[:-1]: num[c] = m[c].fillna(4).values / 7
    num["companions"] = m.TRAVEL_COMPANIONS_NUM.fillna(0).clip(0, 6).values / 6
    return users, sggs, U[KR_CAT].values.astype("int64"), num.values.astype("float32"), Y, S

# ══ 도메인 B: 외국인 (파인튜닝) ══════════════════════════════════
FR_CAT = ["nat", "age", "revisit", "fit", "purpose"]
FR_NUM = ["stay", "nvisit", "companions"]
NAT = {1:"중국",2:"일본",3:"대만",4:"미국",5:"홍콩",6:"태국",7:"베트남",8:"말레이시아",9:"필리핀",
 10:"싱가포르",11:"러시아",12:"중동",13:"인도네시아",14:"캐나다",15:"호주",16:"영국",17:"몽골",
 18:"독일",19:"프랑스",20:"인도",97:"기타"}

def foreign_table():
    """반환: 응답자×시도 롱포맷 (라벨 해상도가 시도)"""
    df = D.survey(); vis = D.survey_visit_cols(df)
    regs = [r for r in C.GATEWAY_REG + C.LOCAL12 if r in vis]
    natn = df["국가별"].map(NAT).fillna("기타")
    U = pd.DataFrame(index=df.index)
    U["nat"] = pd.Categorical(natn, categories=sorted(NAT.values())).codes.clip(0)
    U["age"] = (df["연령별"] - 1).clip(0, 5).fillna(2).astype(int)
    U["revisit"] = (df["방한횟수별"] >= 2).astype(int)
    U["fit"] = (df["여행형태별"] == 1).astype(int)
    U["purpose"] = (df["방한목적별"] - 1).clip(0, 4).fillna(0).astype(int)
    num = pd.DataFrame(index=df.index)
    num["stay"] = pd.to_numeric(df["문9-3. 총 체재기간"], errors="coerce").fillna(0).clip(0, 30) / 30
    num["nvisit"] = df["방한횟수별"].fillna(1).clip(1, 4) / 4
    num["companions"] = df["문7-1. 동반자 수(본인포함)(평균)"].fillna(1).clip(0, 6) / 6
    Y = np.stack([df[vis[r]].notna().astype("float32").values for r in regs], 1)
    gw = D.gateway_vars()
    acc = np.zeros((len(df), len(regs), 2), "float32")
    for j, r in enumerate(regs):
        s = [gw.loc[(n, r)] if (n, r) in gw.index else pd.Series({"supply": 0., "moj_share": 0.})
             for n in natn]
        acc[:, j, 0] = np.log1p([x["supply"] for x in s])
        acc[:, j, 1] = [x["moj_share"] for x in s]
    w = pd.to_numeric(df["1~4분기 가중치"], errors="coerce").fillna(1).values.astype("float32")
    return regs, U[FR_CAT].values.astype("int64"), num.values.astype("float32"), Y, acc, w

def user_split(n, ratios=C.HP.split, seed=C.HP.seed):
    rng = np.random.default_rng(seed); p = rng.permutation(n)
    k = int(n * ratios[0]); s = np.ones(n, int); s[p[:k]] = 0
    return s
