#!/usr/bin/env python3
"""Proty V2 evaluation tuner — Texel-style, Anciency's method, modern
elite data.

Positions come from tools/extract_v2.py (66-byte records: piece
placement, side to move, game result from the side-to-move point of
view) and are labelled ONLY by the game result. The evaluation
parameters are fitted so sigmoid(eval / SCALE) matches the result in
mean-square error — the same loss, parameterisation and optimiser
shape as Anciency's tools/train.py:

  * feature layout identical to src/eval.rs (White-perspective eval;
    records are converted from side-to-move labels to White labels,
    which is an exact sign symmetry of this model);
  * PST-only reparameterisation (the value/PST split is exactly
    collinear, so values are pinned to 0 and each PST absorbs its
    piece's base value; start tables are the constant material values
    100/300/300/500/900/0);
  * ridge least-squares start on the linearised target, then a
    full-batch Adam polish with the midgame/endgame split penalty
    kept on, best checkpoint by training loss with max split <= 220;
  * losses reported for the crude start, the handcrafted BASELINE
    (parsed out of src/tuned.rs) and the tuned set, on both the
    training split and the held-out validation split.

Outputs: src/trained_params.inc, data/theta_trained.npy,
data/training_report.json.
"""
import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack

ROOT = Path(__file__).resolve().parent.parent
SCALE = 200.0
NV, NPST, BLOCK = 6, 384, 391
N_PARAMS = BLOCK * 2
BASE_VALUES = [100, 300, 300, 500, 900, 0]
PHASE_W = np.array([0, 1, 1, 2, 4, 0], dtype=np.float64)


def load_records(pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        raise SystemExit(f"no records match {pattern}")
    parts = []
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        assert raw.size % 66 == 0, f"{f}: size {raw.size} not multiple of 66"
        parts.append(raw.reshape(-1, 66))
    return np.concatenate(parts)


def build_features(rec, chunk=400_000):
    """Records -> (Xmg, Xeg) CSR float32 (N x 391 each), y (White POV)."""
    n = len(rec)
    y = np.where(rec[:, 64] == 0, rec[:, 65] / 2.0, 1.0 - rec[:, 65] / 2.0)
    xmgs, xegs = [], []
    for lo in range(0, n, chunk):
        r = rec[lo:lo + chunk]
        m = len(r)
        sq = r[:, :64]
        flat = sq.ravel()
        mask = flat > 0
        pos_idx = np.repeat(np.arange(m, dtype=np.int32), 64)[mask]
        code = flat[mask].astype(np.int32)
        sq_idx = np.tile(np.arange(64, dtype=np.int32), m)[mask]
        role = (code - 1) % 6
        is_w = code <= 6
        sign = np.where(is_w, 1.0, -1.0).astype(np.float32)
        pst = np.where(is_w, sq_idx ^ 56, sq_idx)
        phase = np.bincount(pos_idx, weights=PHASE_W[role],
                            minlength=m).clip(max=24)
        w_mg = (phase / 24.0).astype(np.float32)
        w_eg = (1.0 - phase / 24.0).astype(np.float32)
        pmg = w_mg[pos_idx]
        peg = w_eg[pos_idx]
        pst_col = 6 + role * 64 + pst
        rows = [pos_idx, pos_idx]
        bw = np.where((sq == 3).sum(axis=1) >= 2)[0]
        bb = np.where((sq == 9).sum(axis=1) >= 2)[0]
        for target_list, w in ((xmgs, pmg), (xegs, peg)):
            cols = np.concatenate([role, pst_col,
                                   np.full(len(bw) + len(bb), NV + NPST,
                                           dtype=np.int32)])
            data = np.concatenate([sign * w, sign * w,
                                   w_mg[bw] if w is pmg else w_eg[bw],
                                   -w_mg[bb] if w is pmg else -w_eg[bb]])
            rr = np.concatenate([rows[0], rows[1],
                                 bw.astype(np.int32), bb.astype(np.int32)])
            target_list.append(csr_matrix((data, (rr, cols)), shape=(m, BLOCK)))
        del flat, mask, pos_idx, code, sq_idx
    Xmg = vstack(xmgs, format="csr")
    Xeg = vstack(xegs, format="csr")
    return Xmg, Xeg, y


def baseline_vector():
    """Parse the BASELINE constant out of src/tuned.rs."""
    text = (ROOT / "src" / "tuned.rs").read_text()
    region = text.split("pub const BASELINE")[1].split("// TRAINED")[0]

    def arr(name):
        m = re.search(name + r": \[([^\]]*)\]", region)
        return [int(x) for x in m.group(1).replace("\n", " ").split(",")
                if x.strip()]

    def pst(name):
        m = re.search(name + r": \[(.*?)\n    \],", region, re.S)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        nums = [int(x) for x in re.findall(r"-?\d+", body)]
        assert len(nums) == 384, (name, len(nums))
        return nums

    v = np.zeros(N_PARAMS)
    v[0:6] = arr("value_mg")
    v[BLOCK:BLOCK + 6] = arr("value_eg")
    v[6:6 + 384] = pst("pst_mg")
    v[BLOCK + 6:BLOCK + 6 + 384] = pst("pst_eg")
    v[NV + NPST] = int(re.search(r"bishop_pair_mg: (-?\d+)", region).group(1))
    v[BLOCK + NV + NPST] = int(re.search(r"bishop_pair_eg: (-?\d+)",
                                  region).group(1))
    return v


def make_loss(Xmg, Xeg, y):
    def loss(theta):
        e = Xmg @ theta[:BLOCK] + Xeg @ theta[BLOCK:]
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        return float(np.mean((p - y) ** 2))
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--train-cap", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    rec_train = load_records("data/positions/shard*-train.bin")
    rec_val = load_records("data/positions/shard*-val.bin")
    if args.train_cap and len(rec_train) > args.train_cap:
        rec_train = rec_train[:args.train_cap]
    print(f"records: {len(rec_train)} train, {len(rec_val)} val "
          f"({time.time()-t0:.0f}s)", flush=True)
    Xmg, Xeg, y = build_features(rec_train)
    Vmg, Veg, vy = build_features(rec_val)
    print(f"features built: train nnz {Xmg.nnz + Xeg.nnz}, "
          f"val nnz {Vmg.nnz + Veg.nnz} ({time.time()-t0:.0f}s)", flush=True)
    train_loss = make_loss(Xmg, Xeg, y)
    val_loss = make_loss(Vmg, Veg, vy)

    # Free parameters: PSTs + bishop pairs (values pinned to 0).
    free = np.array([c for c in range(N_PARAMS)
                     if not (0 <= c < NV or BLOCK <= c < BLOCK + NV)])
    pos_of = {c: i for i, c in enumerate(free)}
    pairs = [(pos_of[NV + k], pos_of[BLOCK + NV + k]) for k in range(NPST + 1)]
    pi = np.array([a for a, _ in pairs])
    pj = np.array([b for _, b in pairs])

    theta0 = np.zeros(N_PARAMS)  # crude start in full space
    for ri, val in enumerate(BASE_VALUES):
        theta0[NV + ri * 64: NV + (ri + 1) * 64] = val
        theta0[BLOCK + NV + ri * 64: BLOCK + NV + (ri + 1) * 64] = val
    hand = baseline_vector()
    print(f"crude start:  train {train_loss(theta0):.6f}  "
          f"val {val_loss(theta0):.6f}", flush=True)
    print(f"handcrafted:  train {train_loss(hand):.6f}  "
          f"val {val_loss(hand):.6f}", flush=True)

    # --- Ridge least-squares start on the linearised target ----------
    start_p = theta0[free].copy()
    target = (y - 0.5) * (4.0 * SCALE)
    gram = np.zeros((len(free), len(free)))
    rhs = np.zeros(len(free))
    CH = 100_000
    for lo in range(0, len(y), CH):
        hi = min(lo + CH, len(y))
        a = Xmg[lo:hi].toarray().astype(np.float64)  # mg block, tapered
        b = Xeg[lo:hi].toarray()
        Xfull = np.concatenate([a, b], axis=1)[:, free]
        resid = target[lo:hi] - Xfull @ start_p
        gram += Xfull.T @ Xfull
        rhs += Xfull.T @ resid
        del a, b, Xfull
    SPLIT_REL = 0.05
    lam = SPLIT_REL * float(np.mean(np.diag(gram)))
    D = np.zeros_like(gram)
    for i, j in pairs:
        D[i, i] += lam
        D[j, j] += lam
        D[i, j] -= lam
        D[j, i] -= lam
    ridge = 1e-3 * np.trace(gram) / gram.shape[0]
    delta = np.linalg.solve(gram + ridge * np.eye(len(free)) + D, rhs)
    theta_p = np.clip(start_p + delta, -800, 800)
    theta_ls = np.zeros(N_PARAMS)
    theta_ls[free] = theta_p
    print(f"after ridge LS: train {train_loss(theta_ls):.6f}  "
          f"val {val_loss(theta_ls):.6f} ({time.time()-t0:.0f}s)", flush=True)

    # --- Adam polish (split penalty on, best checkpoint kept) --------
    # Gradient needs X over free columns only; slice both blocks once.
    free_mg = np.array([c for c in free if c < BLOCK])
    free_eg = np.array([c - BLOCK for c in free if c >= BLOCK])
    Amg = Xmg[:, free_mg].tocsr()
    Aeg = Xeg[:, free_eg].tocsr()
    idx_mg = np.array([pos_of[c] for c in free if c < BLOCK])
    idx_eg = np.array([pos_of[c] for c in free if c >= BLOCK])

    def full_from_p(tp):
        th = np.zeros(N_PARAMS)
        th[free] = tp
        return th

    def data_loss_p(tp):
        e = Amg @ tp[idx_mg] + Aeg @ tp[idx_eg]
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        return float(np.mean((p - y) ** 2))

    MU_SPLIT = 2e-6
    cur = data_loss_p(theta_p)
    best_p, best_loss = theta_p.copy(), cur
    m_ad = np.zeros_like(theta_p)
    v_ad = np.zeros_like(theta_p)
    print(f"Adam polish: start loss {cur:.6f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        e = Amg @ theta_p[idx_mg] + Aeg @ theta_p[idx_eg]
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        diff = p - y
        coeff = (2.0 / len(y)) * diff * p * (1.0 - p) / SCALE
        grad = np.zeros_like(theta_p)
        grad[idx_mg] = Amg.T @ coeff
        grad[idx_eg] = Aeg.T @ coeff
        pen_diff = theta_p[pi] - theta_p[pj]
        grad[pi] += MU_SPLIT * pen_diff
        grad[pj] -= MU_SPLIT * pen_diff
        m_ad = 0.9 * m_ad + 0.1 * grad
        v_ad = 0.999 * v_ad + 0.001 * grad * grad
        m_hat = m_ad / (1 - 0.9 ** epoch)
        v_hat = v_ad / (1 - 0.999 ** epoch)
        lr = 1.0 if epoch <= int(args.epochs * 0.7) else 0.4
        theta_p = theta_p - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        theta_p = np.clip(theta_p, -800, 800)
        if epoch % 25 == 0:
            split_now = float(np.max(np.abs(theta_p[pi] - theta_p[pj])))
            l_now = data_loss_p(theta_p)
            if l_now < best_loss and split_now <= 220:
                best_loss, best_p = l_now, theta_p.copy()
            if epoch % 100 == 0:
                print(f"  epoch {epoch}: loss {l_now:.6f} "
                      f"(best {best_loss:.6f}), max split {split_now:.0f}, "
                      f"{time.time()-t0:.0f}s", flush=True)
    theta_p = best_p
    theta = full_from_p(theta_p)
    print(f"tuned: train {train_loss(theta):.6f}  val {val_loss(theta):.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    np.save(ROOT / "data" / "theta_trained.npy", theta)

    def fmt_arr(vals):
        return "[" + ", ".join(str(int(round(v))) for v in vals) + "]"

    def fmt_pst(block):
        lines = []
        for ri in range(6):
            table = block[ri * 64:(ri + 1) * 64]
            lines.append("        [" + ", ".join(
                str(int(round(v))) for v in table) + "]")
        return "[\n" + ",\n".join(lines) + "\n    ]"

    mg, eg = theta[:BLOCK], theta[BLOCK:]
    inc = f"""EvalParams {{
    value_mg: {fmt_arr(mg[0:6])},
    value_eg: {fmt_arr(eg[0:6])},
    pst_mg: {fmt_pst(mg[6:6 + 384])},
    pst_eg: {fmt_pst(eg[6:6 + 384])},
    bishop_pair_mg: {int(round(mg[NV + NPST]))},
    bishop_pair_eg: {int(round(eg[NV + NPST]))},
}}"""
    (ROOT / "src" / "trained_params.inc").write_text(inc + "\n")

    names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    report = {
        "train_positions": int(len(y)),
        "val_positions": int(len(vy)),
        "scale": SCALE,
        "skip_plies": 6,
        "material_cap_cp": 1000,
        "per_game_cap": 16,
        "parameterization": "PST-only; value_* emitted as 0, each PST "
                            "carries its piece's base value",
        "loss_crude_start": {"train": train_loss(theta0),
                             "val": val_loss(theta0)},
        "loss_handcrafted_baseline": {"train": train_loss(hand),
                                      "val": val_loss(hand)},
        "loss_ridge_ls": {"train": train_loss(theta_ls),
                          "val": val_loss(theta_ls)},
        "loss_tuned": {"train": train_loss(theta), "val": val_loss(theta)},
        "implied_value_mg_from_pst_mean": {
            names[i]: int(round(float(np.mean(mg[6 + i * 64:6 + (i + 1) * 64]))))
            for i in range(6)},
        "implied_value_eg_from_pst_mean": {
            names[i]: int(round(float(np.mean(eg[6 + i * 64:6 + (i + 1) * 64]))))
            for i in range(6)},
        "bishop_pair_mg": int(round(mg[NV + NPST])),
        "bishop_pair_eg": int(round(eg[NV + NPST])),
        "adam_epochs": args.epochs,
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "data" / "training_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
