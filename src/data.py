"""원천 로딩 + 무거운 집계 캐싱."""
import os, glob, pickle
import numpy as np, pandas as pd
from . import config as C

def _rd(f, **kw):
    for e in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try: return pd.read_csv(f, encoding=e, low_memory=False, **kw)
        except Exception: pass
    raise IOError(f"read fail: {f}")

# ── AI Hub (내국인) ─────────────────────────────────────────────
def aihub_visits() -> pd.DataFrame:
    """전 권역 통합 방문지 테이블. 관광형만, 시군구 키 부여."""
    cache = C.CACHE / "aihub_visits.pkl"
    if cache.exists(): return pd.read_pickle(cache)
    rows = []
    for d in C.AIHUB_DIRS:
        v = _rd(f"{d}/tn_visit_area_info.csv")
        t = _rd(f"{d}/tn_travel.csv")[["TRAVEL_ID", "TRAVELER_ID"]]
        v = v.merge(t, on="TRAVEL_ID", how="left")
        v["src"] = os.path.basename(d)
        rows.append(v)
    V = pd.concat(rows, ignore_index=True)
    V = V[V.VISIT_AREA_TYPE_CD.isin(C.TOUR_TYPES)].copy()
    V["유형"] = V.VISIT_AREA_TYPE_CD.map(C.VISIT_TYPE)
    V["sgg"] = V.ROAD_NM_ADDR.fillna(V.LOTNO_ADDR).map(_sgg_key)
    V = V.dropna(subset=["sgg"])
    V.to_pickle(cache)
    return V

def _sgg_key(a):
    """주소 → '시도 시군구'"""
    if not isinstance(a, str): return None
    p = a.split()
    if len(p) < 2: return None
    sd = C.SIDO_LONG.get(p[0]) or (p[0] if p[0] in C.SIDO17 else None)
    if sd is None: return None
    gu = next((x for x in p[1:3] if x.endswith(("시", "군", "구"))), None)
    return f"{sd} {gu}" if gu else None

def aihub_master() -> pd.DataFrame:
    """전 권역 통합 여행객 마스터."""
    rows = []
    for d in C.AIHUB_DIRS:
        m = _rd(f"{d}/tn_traveller_master.csv"); m["src"] = os.path.basename(d)
        rows.append(m)
    return pd.concat(rows, ignore_index=True).drop_duplicates("TRAVELER_ID")

# ── 외국인 조사 ─────────────────────────────────────────────────
def survey() -> pd.DataFrame:
    df = _rd(C.SURVEY_CSV)
    df["pid"] = np.arange(len(df))
    return df

def survey_visit_cols(df) -> dict:
    return {c.split(". ")[-1].rstrip(")"): c
            for c in df.columns if c.startswith("문9-2. 방문 지역(")}

def gateway_vars() -> pd.DataFrame:
    g = _rd(C.GATEWAY_CSV, usecols=["nat", "region", "supply", "moj_share"])
    return g.drop_duplicates(["nat", "region"]).set_index(["nat", "region"])

# ── 카드 (시군구 소비 프로필) ────────────────────────────────────
def card_profiles(level="gu") -> pd.DataFrame:
    p = C.CACHE / f"card_profile_{level}.pkl"
    if p.exists():
        d = pd.read_pickle(p)
        if level == "gu": return d
        return d
    raise FileNotFoundError(f"{p} 없음 — project1의 artifacts에서 복사하세요")

# ── 공공데이터 (아이템 보조 피처) ───────────────────────────────
def entrance_stats() -> pd.DataFrame:
    f = glob.glob(str(C.DATA / "공공데이터" / "입장객통계_*관광지별*.csv"))
    if not f: return pd.DataFrame()
    e = _rd(f[0])
    e["sgg"] = e.시도.map(lambda s: C.SIDO_LONG.get(s, s)) + " " + e.군구.astype(str)
    g = e.groupby("sgg")
    return pd.DataFrame({
        "ent_frgn": np.log1p(g.외국인.sum()), "ent_tot": np.log1p(g.합계.sum()),
        "ent_ratio": (g.외국인.sum() / (g.합계.sum() + 1)), "ent_sites": np.log1p(g.size()),
    })

def resource_stats() -> pd.DataFrame:
    f = glob.glob(str(C.DATA / "공공데이터" / "관광지정보_시군구집계*.csv"))
    if not f: return pd.DataFrame()
    r = _rd(f[0])
    r["sgg"] = r.시도약칭.astype(str) + " " + r.군구.astype(str)
    return r.set_index("sgg")[["관광지수", "숙박", "레저", "휴양문화"]].apply(np.log1p)
