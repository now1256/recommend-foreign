"""경로·상수·하이퍼파라미터. PROJECT_ROOT 기준 상대경로라 폴더째 옮겨도 동작한다."""
import os, glob, unicodedata
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA  = Path(os.environ.get("P2_DATA",  PROJECT_ROOT / "data"))
CACHE = Path(os.environ.get("P2_CACHE", PROJECT_ROOT / "artifacts"))
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AIHUB_DIRS = sorted(glob.glob(str(DATA / "aihub" / "*")))     # 권역_split 폴더들
SURVEY_CSV = next(iter(glob.glob(str(DATA / "02_외래관광객조사" / "*renamed*.csv"))), None)
CODEBOOK   = next((p for p in glob.glob(str(DATA / "02_외래관광객조사" / "*.xlsx"))
                   if "DATA" not in os.path.basename(p)), None)
GATEWAY_CSV = DATA / "05_가공데이터" / "logit_dataset.csv"

# ── 지역 ────────────────────────────────────────────────────────
CAPITAL = ["서울", "경기", "인천"]
GATEWAY_REG = ["부산", "제주"]
LOCAL12 = ["강원","충북","충남","세종","대전","전북","전남","광주","경북","대구","경남","울산"]
SIDO17 = CAPITAL + GATEWAY_REG + LOCAL12
SIDO_LONG = {"서울특별시":"서울","부산광역시":"부산","대구광역시":"대구","인천광역시":"인천",
  "광주광역시":"광주","대전광역시":"대전","울산광역시":"울산","세종특별자치시":"세종",
  "경기도":"경기","강원특별자치도":"강원","강원도":"강원","충청북도":"충북","충청남도":"충남",
  "전북특별자치도":"전북","전라북도":"전북","전라남도":"전남","경상북도":"경북",
  "경상남도":"경남","제주특별자치도":"제주"}

# ── AI Hub 방문지 유형 (VIS 코드) ────────────────────────────────
VISIT_TYPE = {1:"자연",2:"역사",3:"문화",4:"상업",5:"레저",6:"테마",7:"산책",8:"축제",13:"체험"}
TOUR_TYPES = tuple(VISIT_TYPE)          # 관광형만 (집·숙소·역 제외)

class HP:
    dim = 128            # 잠재 차원
    emb_k = 12           # 범주형 임베딩 차원
    d_in = 96            # 어댑터 출력 폭 (도메인 공통)
    hidden = 128
    dropout = 0.1
    lr_pre = 2e-3        # 사전학습 학습률
    lr_ft  = 5e-4        # 파인튜닝 학습률 (낮게)
    weight_decay = 1e-5
    batch = 4096
    epochs_pre = 30
    epochs_ft = 20
    sat_weight = 0.3     # 만족도 보조과제 가중치
    seed = 0
    split = (0.75, 0.25) # train / test (사용자 단위)
