"""사전학습(내국인) → 파인튜닝(외국인) 전이 모델.

설계 3원칙
 1) 입력 어댑터만 도메인별. 의미를 담는 trunk·item은 공유·전이된다.
 2) 아이템에 ID 임베딩을 쓰지 않는다 (피처만) → 학습에 없던 지역도 점수가 나온다.
 3) 파인튜닝 때 item tower를 동결한다 → 내국인이 만든 시군구 공간을 얇은 외국인 데이터가 뭉개지 않게.
"""
import numpy as np, torch, torch.nn as nn
from . import config as C

class InputAdapter(nn.Module):
    """도메인 고유 입력(스키마 다름) → 공통 폭 d_in"""
    def __init__(self, cat_sizes, n_num, d_in=C.HP.d_in, k=C.HP.emb_k):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(v, k) for v in cat_sizes])
        self.proj = nn.Linear(k * len(cat_sizes) + n_num, d_in)
    def forward(self, xc, xn):
        e = torch.cat([self.emb[i](xc[:, i]) for i in range(xc.shape[1])], 1)
        return torch.relu(self.proj(torch.cat([e, xn], 1)))

class TransferRec(nn.Module):
    def __init__(self, kr_cat, kr_num, fr_cat, fr_num, n_item, n_acc=2,
                 dim=C.HP.dim, hidden=C.HP.hidden, dropout=C.HP.dropout):
        super().__init__()
        self.ad_kr = InputAdapter(kr_cat, kr_num)          # 내국인 어댑터
        self.ad_fr = InputAdapter(fr_cat, fr_num)          # 외국인 어댑터
        self.trunk = nn.Sequential(                        # ★공유·전이★
            nn.Linear(C.HP.d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.item = nn.Sequential(                         # ★사전학습 후 동결★
            nn.Linear(n_item, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.sat = nn.Linear(dim, 1)                       # 만족도 보조과제 헤드
        self.acc = nn.Linear(n_acc, 1)                     # 외국인 접근성 게이트
        self.dim = dim

    def user_vec(self, xc, xn, domain):
        h = self.ad_kr(xc, xn) if domain == "kr" else self.ad_fr(xc, xn)
        return self.trunk(h)

    def forward(self, xc, xn, xi, domain, xacc=None):
        z = self.user_vec(xc, xn, domain); it = self.item(xi)
        inter = z * it
        s = inter.sum(1) / np.sqrt(self.dim)
        if xacc is not None: s = s + self.acc(xacc).squeeze(-1)
        return s, self.sat(inter).squeeze(-1)

    # ── 동결 제어 ────────────────────────────────────────────────
    def freeze_for_finetune(self, unfreeze_trunk=False):
        for p in self.parameters(): p.requires_grad = True
        for p in self.item.parameters(): p.requires_grad = False    # 시군구 공간 보존
        for p in self.ad_kr.parameters(): p.requires_grad = False   # 내국인 어댑터 불필요
        if not unfreeze_trunk:
            for p in self.trunk.parameters(): p.requires_grad = False
    def trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
