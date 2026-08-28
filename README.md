# project2 — 외국인 지방 관광 추천

2026 문화체육관광 통계 활용대회. `project`(1차)의 한계였던 **시군구 단위 라벨 부재**를
AI Hub 국내 여행로그(내국인)로 우회한 2차 시도.

## 문서 (읽는 순서)

| 문서 | 내용 |
|---|---|
| **[PIPELINE.md](PIPELINE.md)** | 3단 파이프라인 설계 · 각 단의 검증 가능성 ← **먼저** |
| **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** | 실험 8건 기록 · 무엇이 왜 실패/성공했나 |
| [DATASETS.md](DATASETS.md) | 데이터 출처·형상·한계 |
| [DATA_SETUP.md](DATA_SETUP.md) | 데이터 내려받기 (git에 미포함) |

## 결론 한 줄

> **"이 사람이 누구인가"(인구·성향)에는 신호가 없고,
> "이 사람이 어디를 갔는가"(행동)에는 신호가 있다.**

개인화 추천은 8번의 실험에서 모두 실패했고(리프트 상한 1.10배),
행동 기반 연관 추천(EASE)이 인기순의 **4.5배**(HR@10)로 작동한다.

## 성능

| 단계 | 모델 | 지표 | 대조군 |
|---|---|---|---|
| 1단 시도 추천 | Two-Tower | R@1 **0.466** | 인기순 0.359 |
| 2·3단 POI 추천 | **EASE** (λ=500) | HR@1 **0.200** / HR@10 **0.678** | 인기순 0.151 |

⚠️ **1단과 2·3단은 서로 다른 데이터로 검증됐다.** 통합 성능은 검증 불가 — [PIPELINE.md](PIPELINE.md) 참조.

## 빠른 시작

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 데이터 준비: DATA_SETUP.md 참조
python -c "
import sys; sys.path.insert(0,'.')
from src import data as D
V=D.aihub_visits(); print(f'방문 {len(V):,} / 여행객 {V.TRAVELER_ID.nunique():,}')"
```

## 데이터 현황

| 데이터 | 규모 |
|---|---|
| AI Hub 여행로그 (내국인) | 여행객 **9,269명** / 방문 **33,895건** / POI 11,326개 / 시군구 227개 |
| 외래관광객조사 (외국인) | 16,110명 × 408변수 |
| 신한카드 (외국인 소비) | 시군구 205개 집계 캐시 |
| 공공데이터 | 입장객·관광지정보 |

권역: 수도권·서부권·동부권 (1·2차년도). **제주 미확보.**

## 폴더

```
project2/
├── PIPELINE.md / EXPERIMENT_LOG.md / DATASETS.md / DATA_SETUP.md
├── src/
│   ├── config.py    경로·상수 (PROJECT_ROOT 기준 상대경로)
│   ├── data.py      AI Hub 권역 통합 · 주소→시군구 · 캐싱
│   ├── features.py  학습표 구성
│   ├── models.py    TransferRec (사전학습→파인튜닝 실험용)
│   ├── train.py     pretrain / finetune
│   └── evaluate.py  Recall@k · popularity · coverage
├── data/            (.gitignore) — DATA_SETUP.md 참조
└── artifacts/       캐시 · 체크포인트
```

## 1차 프로젝트와의 관계

| | project (1차) | project2 (2차) |
|---|---|---|
| 접근 | 외국인 조사만으로 시도 추천 | 내국인 데이터로 POI 해상도 확보 |
| 결과 | R@1 0.466에서 정체 | 개인화 실패 확인 · **CF로 전환 성공** |
| 남긴 것 | Two-Tower (1단으로 계속 사용) | EASE (2·3단) |

1차의 *"모델이 아니라 정보의 한계"* 라는 결론을 2차가 **정량적으로 확증**했다(리프트 1.10배).
