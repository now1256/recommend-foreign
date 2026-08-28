"""학습 부족인지 데이터 한계인지 판정.

  python -m src.diagnose --epochs 300

train Recall이 오르는데 test가 안 오르면 → 신호가 일반화 안 됨 (데이터 한계)
train Recall도 안 오르면 → 용량/학습 부족 (에폭·차원을 더 키워볼 것)
"""
import argparse, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from . import config as C, features as F, models as M, evaluate as E

def run(epochs=300, dim=None, exclude_capital=False, log_every=25):
    torch.manual_seed(0); np.random.seed(0)
    if dim: C.HP.dim = dim
    users, sggs, UC, UN, Y, S = F.korean_table()
    if exclude_capital:
        k = np.array([s.split()[0] not in C.CAPITAL for s in sggs])
        sggs = [s for s, x in zip(sggs, k) if x]; Y, S = Y[:, k], S[:, k]
        h = Y.sum(1) > 0; UC, UN, Y, S = UC[h], UN[h], Y[h], S[h]
    IM = F.item_matrix(sggs).values.astype("float32")
    nU, nG = len(UC), len(sggs); part = F.user_split(nU)
    net = M.TransferRec([int(UC[:, i].max())+1 for i in range(UC.shape[1])], UN.shape[1],
                        [21,6,2,2,5], 3, IM.shape[1]).to(C.DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=C.HP.lr_pre, weight_decay=C.HP.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    uu = np.repeat(np.arange(nU), nG); gg = np.tile(np.arange(nG), nU); tr = part[uu] == 0
    T = lambda a: torch.tensor(a, device=C.DEVICE)
    Xc, Xn, Xi = T(UC[uu][tr]), T(UN[uu][tr]), T(IM[gg][tr])
    yy, ss = T(Y.reshape(-1)[tr]), T(S.reshape(-1)[tr])
    Ac, An, Ai = T(UC[uu]), T(UN[uu]), T(IM[gg])
    pop_tr = E.popularity_baseline(Y[part==0], Y[part==0])["R@3"]
    pop_te = E.popularity_baseline(Y[part==0], Y[part==1])["R@3"]
    print(f"n_user={nU} n_item={nG} pos={int(Y.sum())} ({Y.mean()*100:.2f}%) dim={C.HP.dim} device={C.DEVICE}")
    print(f"인기순 baseline  train R@3={pop_tr:.4f}  test R@3={pop_te:.4f}\n")
    print(f"{'ep':>4} {'loss':>8} {'train R@3':>10} {'test R@3':>9} {'top1종류':>8} {'초':>6}")
    t0 = time.time(); hist = []
    for ep in range(1, epochs+1):
        net.train(); pm = torch.randperm(len(yy), device=C.DEVICE)
        for i in range(0, len(pm), C.HP.batch):
            b = pm[i:i+C.HP.batch]; opt.zero_grad()
            o, sp = net(Xc[b], Xn[b], Xi[b], "kr"); l = bce(o, yy[b])
            m = ~torch.isnan(ss[b])
            if m.any(): l = l + C.HP.sat_weight*nn.functional.mse_loss(sp[m], ss[b][m]/5.0)
            l.backward(); opt.step()
        if ep % log_every == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                sc = net(Ac, An, Ai, "kr")[0].cpu().numpy().reshape(nU, nG)
            rtr = E.recall_at_k(sc[part==0], Y[part==0])["R@3"]
            rte = E.recall_at_k(sc[part==1], Y[part==1])["R@3"]
            cov = E.coverage(sc[part==1], sggs, 1)["n_items_topk"]
            print(f"{ep:4d} {l.item():8.4f} {rtr:10.4f} {rte:9.4f} {cov:8d} {time.time()-t0:6.0f}")
            hist.append((ep, rtr, rte))
    print("\n[판정]")
    _, rtr, rte = hist[-1]
    if rtr > pop_tr + 0.05:
        print(f"  train R@3 {rtr:.4f} > 인기순 {pop_tr:.4f} → 모델은 학습 데이터를 맞힐 수 있다.")
        print(f"  test  R@3 {rte:.4f} vs 인기순 {pop_te:.4f} → " +
              ("일반화됨 (신호 있음)" if rte > pop_te + 0.02 else "일반화 실패 = 데이터 한계"))
    else:
        print(f"  train R@3 {rtr:.4f} ≈ 인기순 {pop_tr:.4f} → 학습 데이터조차 못 맞힘.")
        print("  → 용량/에폭 부족 가능. --dim 256 --epochs 1000 으로 재시도 권장")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--exclude-capital", action="store_true")
    a = ap.parse_args()
    run(a.epochs, a.dim, a.exclude_capital)
