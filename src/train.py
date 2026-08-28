"""학습 엔트리포인트.

  python -m src.train pretrain              # 1단계: 내국인 사전학습
  python -m src.train finetune              # 2단계: 외국인 파인튜닝 (사전학습 필요)
  python -m src.train baseline              # 대조군: 사전학습 없이 외국인만
  python -m src.train all                   # 전부 순서대로
"""
import argparse, time, json
import numpy as np, torch, torch.nn as nn
from . import config as C, features as F, models as M, evaluate as E

def _t(x): return torch.tensor(x, device=C.DEVICE)

# ══════════════════════════════════════════════════════════════
def pretrain(epochs=C.HP.epochs_pre, verbose=True):
    """내국인 여행객 × 시군구 방문 (+ 만족도 보조과제)"""
    torch.manual_seed(C.HP.seed); np.random.seed(C.HP.seed)
    users, sggs, UC, UN, Y, S = F.korean_table()
    IM = F.item_matrix(sggs).values.astype("float32")
    part = F.user_split(len(users))                       # 0=train 1=test (사용자 단위)
    nU, nG = len(users), len(sggs)

    model = M.TransferRec([int(UC[:, i].max())+1 for i in range(UC.shape[1])], UN.shape[1],
                          [21, 6, 2, 2, 5], len(F.FR_NUM), IM.shape[1]).to(C.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.HP.lr_pre, weight_decay=C.HP.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    uu = np.repeat(np.arange(nU), nG); gg = np.tile(np.arange(nG), nU)
    tr = part[uu] == 0
    Xc, Xn, Xi = _t(UC[uu][tr]), _t(UN[uu][tr]), _t(IM[gg][tr])
    yy, ss = _t(Y.reshape(-1)[tr]), _t(S.reshape(-1)[tr])
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(len(yy), device=C.DEVICE)
        for i in range(0, len(perm), C.HP.batch):
            b = perm[i:i+C.HP.batch]; opt.zero_grad()
            o, sp = model(Xc[b], Xn[b], Xi[b], "kr")
            loss = bce(o, yy[b])
            m = ~torch.isnan(ss[b])
            if m.any(): loss = loss + C.HP.sat_weight * nn.functional.mse_loss(sp[b][m] if False else sp[m], ss[b][m]/5.0)
            loss.backward(); opt.step()
        if verbose and (ep % 10 == 0 or ep == 1): print(f"  ep{ep:3d} loss {loss.item():.4f}")
    dt = time.time() - t0
    # 평가
    model.eval(); te = part == 1
    with torch.no_grad():
        sc = model(_t(UC[uu]), _t(UN[uu]), _t(IM[gg]), "kr")[0].cpu().numpy().reshape(nU, nG)
    res = E.recall_at_k(sc[te], Y[te]); pop = E.popularity_baseline(Y[part == 0], Y[te])
    cov = E.coverage(sc[te], sggs, k=1)
    torch.save({"state": model.state_dict(), "sggs": sggs,
                "kr_cat": [int(UC[:, i].max())+1 for i in range(UC.shape[1])],
                "kr_num": UN.shape[1], "n_item": IM.shape[1]}, C.CACHE / "pretrained.pt")
    return {"stage": "pretrain", "sec": round(dt, 1), "model": res, "popularity": pop, "coverage": cov,
            "n_users": nU, "n_items": nG, "positives": int(Y.sum())}

# ══════════════════════════════════════════════════════════════
def finetune(use_pretrained=True, unfreeze_trunk=True, epochs=C.HP.epochs_ft, verbose=True):
    """외국인 응답자 × 시도. 사전학습 가중치 전이 여부를 비교한다."""
    torch.manual_seed(C.HP.seed); np.random.seed(C.HP.seed)
    regs, FC, FN, Y, ACC, W = F.foreign_table()
    IM_sido = F.item_matrix(regs, standardize=True)       # 시도 이름으로 아이템 피처 (근사)
    ck = torch.load(C.CACHE / "pretrained.pt", map_location=C.DEVICE, weights_only=False)
    n_item = ck["n_item"]
    IM = np.zeros((len(regs), n_item), "float32")
    common = [c for c in IM_sido.columns][:n_item]
    IM[:, :len(common)] = IM_sido[common].values.astype("float32")

    model = M.TransferRec(ck["kr_cat"], ck["kr_num"], [21, 6, 2, 2, 5], FN.shape[1], n_item).to(C.DEVICE)
    if use_pretrained:
        model.load_state_dict(ck["state"]); model.freeze_for_finetune(unfreeze_trunk)
    nU, nR = Y.shape
    part = F.user_split(nU)
    uu = np.repeat(np.arange(nU), nR); rr = np.tile(np.arange(nR), nU)
    tr = part[uu] == 0
    Xc, Xn, Xi = _t(FC[uu][tr]), _t(FN[uu][tr]), _t(IM[rr][tr])
    Xa = _t(ACC.reshape(-1, 2)[tr]); yy = _t(Y.reshape(-1)[tr]); ww = _t(np.repeat(W, nR)[tr])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=C.HP.lr_ft if use_pretrained else C.HP.lr_pre,
                           weight_decay=C.HP.weight_decay)
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(len(yy), device=C.DEVICE)
        for i in range(0, len(perm), C.HP.batch):
            b = perm[i:i+C.HP.batch]; opt.zero_grad()
            o, _ = model(Xc[b], Xn[b], Xi[b], "fr", Xa[b])
            loss = (nn.functional.binary_cross_entropy_with_logits(o, yy[b], reduction="none")
                    * ww[b] / ww[b].mean()).mean()
            loss.backward(); opt.step()
        if verbose and (ep % 10 == 0 or ep == 1): print(f"  ep{ep:3d} loss {loss.item():.4f}")
    dt = time.time() - t0
    model.eval(); te = part == 1
    with torch.no_grad():
        sc = model(_t(FC[uu]), _t(FN[uu]), _t(IM[rr]), "fr",
                   _t(ACC.reshape(-1, 2)))[0].cpu().numpy().reshape(nU, nR)
    loc = [i for i, r in enumerate(regs) if r in C.LOCAL12]
    out = {"stage": "finetune" if use_pretrained else "baseline", "sec": round(dt, 1),
           "trainable": model.trainable(),
           "all14": E.recall_at_k(sc[te], Y[te]),
           "local12": E.recall_at_k(sc[te][:, loc], Y[te][:, loc]),
           "pop_local12": E.popularity_baseline(Y[part == 0][:, loc], Y[te][:, loc]),
           "coverage12": E.coverage(sc[te][:, loc], [regs[i] for i in loc], k=1)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["pretrain","finetune","baseline","all"])
    a = ap.parse_args(); res = []
    print(f"device={C.DEVICE}")
    if a.stage in ("pretrain", "all"):
        print("\n[1단계] 내국인 사전학습"); r = pretrain(); res.append(r); print(json.dumps(r, ensure_ascii=False, indent=2))
    if a.stage in ("baseline", "all"):
        print("\n[대조군] 사전학습 없이 외국인만"); r = finetune(use_pretrained=False); res.append(r); print(json.dumps(r, ensure_ascii=False, indent=2))
    if a.stage in ("finetune", "all"):
        print("\n[2단계] 외국인 파인튜닝 (전이)"); r = finetune(True, True); res.append(r); print(json.dumps(r, ensure_ascii=False, indent=2))
    (C.CACHE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
