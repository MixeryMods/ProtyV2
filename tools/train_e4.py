#!/usr/bin/env python3
"""Proty V2 Phase B tuner (2.1.0): Texel fit on QUIET positions with
the extended feature set (base material/PST/bishop-pair + 22
structural features per phase block from tools/quiet_features.py).

Same pipeline shape as the 2.0.0 tune (tools/train_v2.py +
tools/lm_polish.py): ridge least-squares start on the linearised
target, then monotone Gauss-Newton (LM) rounds on the true loss.
Labels are game results ONLY. Positions come from
tools/extract_quiet.py (identical game filter and sampling as 2.0.0,
plus the quiet filter: side to move not in check, and the move played
from the position was not a capture, promotion or check).

Losses are reported on the SAME held-out quiet validation set for:
the handcrafted BASELINE, the 2.0.0 tuned parameter set (old features,
extras zero) and the new tuned set.

MEMORY: this VM has ~3.6 GB available and no swap. Feature matrices
are therefore built per corpus shard, persisted to .npz, and every
gram/loss pass streams shard by shard; no full-width dense copy and
no full-matrix column slice is ever materialised.

Outputs: src/trained_params.inc (extended format), data/theta_v3.npy,
data/training_report_e4.json.
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz, vstack

import importlib.util
spec = importlib.util.spec_from_file_location(
    "tv2", Path(__file__).resolve().parent / "train_v2.py")
tv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv2)

ROOT = tv2.ROOT
SCALE = tv2.SCALE
BASE_BLOCK = tv2.BLOCK          # 391
EXTRA = 22
BLOCK = BASE_BLOCK + EXTRA      # 413
N_PARAMS = BLOCK * 2            # 826
PHASE_W = tv2.PHASE_W
FEAT_DIR = ROOT / "data" / "e4_2200"


def load_extra(pattern, expect_rows):
    files = sorted(glob.glob(str(ROOT / pattern)))
    assert files, pattern
    parts = [np.load(f) for f in files]
    ftr = np.concatenate(parts)
    assert len(ftr) == expect_rows, (len(ftr), expect_rows)
    assert ftr.shape[1] == EXTRA
    return ftr


def compute_phase(rec, chunk=250_000):
    """Game phase (0..24) per record, in row chunks with small
    dtypes — a naive full-width float64 pass over (N x 64) records
    OOM-kills this VM."""
    n = len(rec)
    phase = np.zeros(n, dtype=np.float64)
    for lo in range(0, n, chunk):
        sq = rec[lo:lo + chunk, :64].astype(np.int16)
        role = np.where(sq > 0, (sq - 1) % 6, 0)
        w = PHASE_W[role] * (sq > 0)
        phase[lo:lo + chunk] = w.sum(axis=1)
    return np.clip(phase, 0, 24)


def extra_csr(ftr, w, chunk=250_000):
    parts = []
    for lo in range(0, len(ftr), chunk):
        block = (ftr[lo:lo + chunk] * w[lo:lo + chunk, None]).astype(
            np.float32)
        parts.append(csr_matrix(block))
    return vstack(parts, format="csr")


def build_all(rec_pattern, ftr_pattern):
    rec = tv2.load_records(rec_pattern)
    Xmg, Xeg, y = tv2.build_features(rec)
    ftr = load_extra(ftr_pattern, len(rec)).astype(np.float32)
    phase = compute_phase(rec)
    w_mg = (phase / 24.0).astype(np.float32)
    w_eg = (1.0 - phase / 24.0).astype(np.float32)
    Emg = extra_csr(ftr, w_mg)
    Eeg = extra_csr(ftr, w_eg)
    del rec, ftr
    return (hstack([Xmg, Emg], format="csr"),
            hstack([Xeg, Eeg], format="csr"), y)


def scatter782(vec782):
    """Old-layout (782) parameter vector -> new layout (826)."""
    out = np.zeros(N_PARAMS)
    out[0:BASE_BLOCK] = vec782[0:BASE_BLOCK]
    out[BLOCK:BLOCK + BASE_BLOCK] = vec782[BASE_BLOCK:2 * BASE_BLOCK]
    return out


# ------------------------------------------------------------------
# Streaming dataset: a list of (mg_npz_path, eg_npz_path) shards plus
# the concatenated labels. Every pass loads one shard at a time.
# ------------------------------------------------------------------

def build_dataset(split):
    """Build (or reuse) per-shard feature matrices for 'train'/'val'.
    Returns (shard_paths, y)."""
    shards = []
    ys = []
    for shard in (0, 1):
        mg_p = FEAT_DIR / f"feat-shard{shard}-{split}-mg.npz"
        eg_p = FEAT_DIR / f"feat-shard{shard}-{split}-eg.npz"
        y_p = FEAT_DIR / f"feat-shard{shard}-{split}-y.npy"
        if not (mg_p.exists() and eg_p.exists() and y_p.exists()):
            Xmg, Xeg, y = build_all(
                f"data/e4_2200/eshard{shard}-{split}.bin",
                f"data/e4_2200/eshard{shard}-{split}-ftr.npy")
            save_npz(mg_p, Xmg)
            save_npz(eg_p, Xeg)
            np.save(y_p, y)
            del Xmg, Xeg
        shards.append((mg_p, eg_p))
        ys.append(np.load(y_p))
    return shards, np.concatenate(ys)


def loss_stream(shards, y, theta):
    sse = 0.0
    off = 0
    for mg_p, eg_p in shards:
        Xmg = load_npz(mg_p)
        Xeg = load_npz(eg_p)
        n = Xmg.shape[0]
        e = Xmg @ theta[:BLOCK] + Xeg @ theta[BLOCK:]
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        sse += float(np.sum((p - y[off:off + n]) ** 2))
        off += n
        del Xmg, Xeg
    return sse / len(y)


def stack_free_rows(Xmg, Xeg, free_mg, free_eg, lo, hi):
    """Rows [lo, hi) restricted to the free columns."""
    return hstack([Xmg[lo:hi][:, free_mg], Xeg[lo:hi][:, free_eg]],
                  format="csr")


def main():
    t0 = time.time()
    train_shards, y = build_dataset("train")
    val_shards, vy = build_dataset("val")
    print(f"features ready: {len(y)} train, {len(vy)} val "
          f"({time.time()-t0:.0f}s)", flush=True)

    def train_loss(theta):
        return loss_stream(train_shards, y, theta)

    def val_loss(theta):
        return loss_stream(val_shards, vy, theta)

    free = np.array([c for c in range(N_PARAMS)
                     if not (0 <= c < 6 or BLOCK <= c < BLOCK + 6)])
    pos_of = {c: i for i, c in enumerate(free)}
    pairs = [(pos_of[6 + k], pos_of[BLOCK + 6 + k])
             for k in range(BLOCK - 6)]
    pi = np.array([a for a, _ in pairs])
    pj = np.array([b for _, b in pairs])
    free_mg = np.array([c for c in free if c < BLOCK])
    free_eg = np.array([c - BLOCK for c in free if c >= BLOCK])
    nf = len(free)

    hand = scatter782(tv2.baseline_vector())
    old = scatter782(np.load(ROOT / "data" / "theta_trained.npy"))
    hand_tr, hand_va = train_loss(hand), val_loss(hand)
    old_tr, old_va = train_loss(old), val_loss(old)
    print(f"handcrafted:  train {hand_tr:.6f}  val {hand_va:.6f}",
          flush=True)
    print(f"old tuned:    train {old_tr:.6f}  val {old_va:.6f}",
          flush=True)

    # --- Ridge least-squares start on the linearised target --------
    theta0 = np.zeros(N_PARAMS)
    for ri, val in enumerate(tv2.BASE_VALUES):
        theta0[6 + ri * 64: 6 + (ri + 1) * 64] = val
        theta0[BLOCK + 6 + ri * 64: BLOCK + 6 + (ri + 1) * 64] = val
    start_p = theta0[free].copy()
    target = (y - 0.5) * (4.0 * SCALE)
    gram = np.zeros((nf, nf))
    rhs = np.zeros(nf)
    CH = 200_000
    off = 0
    for mg_p, eg_p in train_shards:
        Xmg = load_npz(mg_p)
        Xeg = load_npz(eg_p)
        n = Xmg.shape[0]
        for lo in range(0, n, CH):
            hi = min(lo + CH, n)
            Xc = stack_free_rows(Xmg, Xeg, free_mg, free_eg, lo, hi)
            resid = target[off + lo:off + hi] - Xc @ start_p
            gram += (Xc.T @ Xc).toarray().astype(np.float64)
            rhs += np.asarray(Xc.T @ resid).ravel()
            del Xc
        off += n
        del Xmg, Xeg
    SPLIT_REL = 0.05
    lam = SPLIT_REL * float(np.mean(np.diag(gram)))
    D = np.zeros_like(gram)
    D[pi, pi] += lam
    D[pj, pj] += lam
    D[pi, pj] -= lam
    D[pj, pi] -= lam
    ridge = 1e-3 * np.trace(gram) / gram.shape[0]
    delta = np.linalg.solve(gram + ridge * np.eye(nf) + D, rhs)
    del gram, rhs, D
    theta_ls = np.zeros(N_PARAMS)
    theta_ls[free] = np.clip(start_p + delta, -800, 800)
    ls_tr, ls_va = train_loss(theta_ls), val_loss(theta_ls)
    print(f"after ridge LS: train {ls_tr:.6f}  val {ls_va:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- Gauss-Newton (LM) polish on the true loss ------------------
    def lm_pass(tp, want_gram):
        """One streaming pass: returns (train_loss, gram, rhs) with
        gram/rhs None unless want_gram."""
        gram = np.zeros((nf, nf)) if want_gram else None
        rhs = np.zeros(nf) if want_gram else None
        sse = 0.0
        off = 0
        for mg_p, eg_p in train_shards:
            Xmg = load_npz(mg_p)
            Xeg = load_npz(eg_p)
            n = Xmg.shape[0]
            for lo in range(0, n, CH):
                hi = min(lo + CH, n)
                Xc = stack_free_rows(Xmg, Xeg, free_mg, free_eg, lo, hi)
                e = Xc @ tp
                p = 1.0 / (1.0 + np.exp(-e / SCALE))
                r = y[off + lo:off + hi] - p
                sse += float(np.sum(r ** 2))
                if want_gram:
                    dp = p * (1.0 - p) / SCALE
                    Xs = Xc.multiply(np.sqrt(dp * dp)[:, None]).tocsr()
                    gram += (Xs.T @ Xs).toarray().astype(np.float64)
                    rhs += np.asarray(Xc.T @ (dp * r)).ravel()
                    del Xs
                del Xc
            off += n
            del Xmg, Xeg
        return sse / len(y), gram, rhs

    def full_losses(tp):
        theta = np.zeros(N_PARAMS)
        theta[free] = tp
        tl, _, _ = lm_pass(tp, False)
        return tl, val_loss(theta)

    tp = theta_ls[free].copy()
    cur, curv = full_losses(tp)
    print(f"LM start: train {cur:.6f} val {curv:.6f}", flush=True)
    for rnd in range(1, 7):
        _, gram, rhs = lm_pass(tp, True)
        md = float(np.mean(np.diag(gram)))
        lam2 = 0.02 * md
        D = np.zeros_like(gram)
        D[pi, pi] += lam2
        D[pj, pj] += lam2
        D[pi, pj] -= lam2
        D[pj, pi] -= lam2
        split_vec = np.zeros(nf)
        split_vec[pi] = tp[pi] - tp[pj]
        split_vec[pj] = -(tp[pi] - tp[pj])
        rhs = rhs - lam2 * split_vec
        step = np.linalg.solve(gram + D + (1e-4 * md) * np.eye(nf), rhs)
        del gram, D
        accepted = False
        for s in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = np.clip(tp + s * step, -800, 800)
            cl, cv = full_losses(cand)
            if cl < cur - 1e-7:
                print(f"round {rnd}: step {s}: train {cl:.6f} val "
                      f"{cv:.6f} ({time.time()-t0:.0f}s)", flush=True)
                tp, cur, curv = cand, cl, cv
                accepted = True
                break
        if not accepted:
            print(f"round {rnd}: no improving step; stop", flush=True)
            break

    theta = np.zeros(N_PARAMS)
    theta[free] = tp
    print(f"tuned: train {cur:.6f}  val {curv:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(ROOT / "data" / "theta_e4.npy", theta)
    emit_inc(theta)
    report = {
        "train_positions": int(len(y)),
        "val_positions": int(len(vy)),
        "scale": SCALE,
        "dataset": "quiet positions (extract_quiet.py): stm not in "
                   "check, move played from the position not a "
                   "capture/promotion/check; same game filter and "
                   "sampling as 2.0.0",
        "features": "base 391/block + 22 structural/block = 413/block",
        "loss_handcrafted_baseline": {"train": hand_tr, "val": hand_va},
        "loss_old_tuned_2_0_0": {"train": old_tr, "val": old_va},
        "loss_ridge_ls": {"train": ls_tr, "val": ls_va},
        "loss_tuned": {"train": cur, "val": curv},
        "extras_tuned": extras_dict(theta),
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "data" / "training_report_e4.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def extras_dict(theta):
    mg, eg = theta[:BLOCK], theta[BLOCK:]
    names = (["passed_r2", "passed_r3", "passed_r4", "passed_r5",
              "passed_r6", "passed_r7", "isolated", "doubled",
              "backward", "shield", "attack_n", "attack_b", "attack_r",
              "attack_q", "mobility_n", "mobility_b", "mobility_r",
              "mobility_q", "rook_open", "rook_semi", "rook_seventh",
              "outpost"])
    return {names[i]: {"mg": int(round(mg[BASE_BLOCK + i])),
                       "eg": int(round(eg[BASE_BLOCK + i]))}
            for i in range(EXTRA)}


def emit_inc(theta):
    mg, eg = theta[:BLOCK], theta[BLOCK:]

    def fmt_arr(vals):
        return "[" + ", ".join(str(int(round(v))) for v in vals) + "]"

    def fmt_pst(block):
        lines = []
        for ri in range(6):
            lines.append("        [" + ", ".join(
                str(int(round(v))) for v in
                block[ri * 64:(ri + 1) * 64]) + "]")
        return "[\n" + ",\n".join(lines) + "\n    ]"

    def fields(b, suffix):
        x = b[BASE_BLOCK:BASE_BLOCK + EXTRA]
        return f"""    passed_{suffix}: {fmt_arr(x[0:6])},
    isolated_{suffix}: {int(round(x[6]))},
    doubled_{suffix}: {int(round(x[7]))},
    backward_{suffix}: {int(round(x[8]))},
    shield_{suffix}: {int(round(x[9]))},
    king_attack_{suffix}: {fmt_arr(x[10:14])},
    mobility_{suffix}: {fmt_arr(x[14:18])},
    rook_open_{suffix}: {int(round(x[18]))},
    rook_semi_{suffix}: {int(round(x[19]))},
    rook_seventh_{suffix}: {int(round(x[20]))},
    outpost_{suffix}: {int(round(x[21]))},"""

    inc = f"""EvalParams {{
    value_mg: {fmt_arr(mg[0:6])},
    value_eg: {fmt_arr(eg[0:6])},
    pst_mg: {fmt_pst(mg[6:6 + 384])},
    pst_eg: {fmt_pst(eg[6:6 + 384])},
    bishop_pair_mg: {int(round(mg[6 + 384]))},
    bishop_pair_eg: {int(round(eg[6 + 384]))},
{fields(mg, "mg")}
{fields(eg, "eg")}
}}"""
    (ROOT / "src" / "trained_params.inc").write_text(inc + "\n")


if __name__ == "__main__":
    main()
