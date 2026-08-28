# 데이터 준비 (git에 포함되지 않음)

`.gitignore`가 `data/`를 제외한다. 클론 후 아래 순서로 채운다.

## 1. AI Hub 국내 여행로그 (내국인)

[AI Hub](https://aihub.or.kr)에서 데이터셋별로 **활용 신청·승인**을 먼저 받아야 한다.

```bash
export AIHUB_KEY='발급받은-API-키'      # 마이페이지 > API 키 발급

# 1차년도 (권역당 약 394MB, 압축 해제 시 5~6GB)
aihubshell -mode d -datasetkey 71581 -filekey 513574 -aihubapikey "$AIHUB_KEY"  # 수도권
aihubshell -mode d -datasetkey 71582 -filekey 512871 -aihubapikey "$AIHUB_KEY"  # 서부권
aihubshell -mode d -datasetkey 71585 -filekey 512841 -aihubapikey "$AIHUB_KEY"  # 동부권
aihubshell -mode l -datasetkey 71584 -aihubapikey "$AIHUB_KEY" | grep -i csv     # 제주 (filekey 확인)

# 2차년도 (권역당 3~5MB, 가벼움)
aihubshell -mode d -datasetkey 71776 -filekey 539784,539787 -aihubapikey "$AIHUB_KEY"  # 수도권
aihubshell -mode d -datasetkey 71778 -filekey 539795,539798 -aihubapikey "$AIHUB_KEY"  # 동부권
aihubshell -mode d -datasetkey 71779 -filekey 539804,539807 -aihubapikey "$AIHUB_KEY"  # 서부권
aihubshell -mode d -datasetkey 71780 -filekey 541667,541670 -aihubapikey "$AIHUB_KEY"  # 제주
```

압축을 풀고 CSV를 아래 구조로 정리한다. **한글 파일명이 깨지므로 ASCII 접두사 기준으로 이름을 바꾼다.**

```
data/aihub/{권역}{연차}_{split}/
    tn_visit_area_info.csv      방문지 (라벨·만족도)
    tn_traveller_master.csv     여행객 특성
    tn_travel.csv               여행
    tc_codeb.csv                코드표
    ... (14개)
```

정리 스크립트 예시는 `scripts/prepare_aihub.sh` 참조.

## 2. 대회 제공 데이터

| 경로 | 내용 |
|---|---|
| `data/02_외래관광객조사/*_renamed.csv` | 외래관광객조사 정제본 (컬럼명 한글화) |
| `data/02_외래관광객조사/*코드북*.xlsx` | 코드북 |
| `data/05_가공데이터/logit_dataset.csv` | 관문 변수(supply·moj_share) |

## 3. 공공데이터

| 파일 | 출처 |
|---|---|
| `data/공공데이터/입장객통계_2025_관광지별.csv` | [관광지식정보시스템](https://know.tour.go.kr) |
| `data/공공데이터/관광지정보_시군구집계.csv` | [공공데이터포털](https://www.data.go.kr/data/15021141/standard.do) |

## 4. 캐시 (선택)

`artifacts/card_profile_gu.pkl` — 신한카드 726MB의 시군구 집계 캐시.
없으면 원본에서 재생성 필요(약 2~4분).

## 확인

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src import data as D
V=D.aihub_visits(); print(f'방문 {len(V):,} / 여행객 {V.TRAVELER_ID.nunique():,} / POI {V.VISIT_AREA_NM.nunique():,}')"
```
