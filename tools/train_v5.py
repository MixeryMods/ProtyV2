#!/usr/bin/env python3
"""Proty V2 v5 tuner (2026-11-01): WARM-START repair of the rejected v4
retune. The from-scratch all-position fit won the val-loss comparison
(0.150933 vs 0.154646) but lost the SPRT gate badly (owner-stopped at
70 games, 30.7%, LLR -1.716): a free refit on result labels wanders
into beliefs that predict game outcomes yet rank the engine's own
moves worse.

This tuner instead starts the Gauss-Newton (LM) polish AT the shipped
0.3.0-0.7.0 parameters (data/theta_v3.npy) and leashes every step with
a proximal penalty toward them:

    objective(tp) = train_loss(tp) + (MU/2) * mean((tp - ref)^2)

so the new data may refine the champion but cannot drag it somewhere
unrecognisable. Same data (cached v4 blocks), same labels (game
results only), same free-parameter set as train_v4.py (piece values
frozen). MU via PROTY_V5_MU (default 1e-5: moving every free param by
~14 cp RMS costs ~0.001 loss, the scale of the gains at stake).

Outputs: data/theta_v5.npy, src/trained_params_v5.inc,
data/training_report_v5.json. Shipped params are NOT touched.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

import numpy as np
from scipy.sparse import load_npz

spec = importlib.util.spec_from_file_location(
    "tv4", Path(__file__).resolve().parent / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv4)

ROOT = tv4.ROOT
SCALE = tv4.SCALE
BLOCK = tv4.BLOCK
N_PARAMS = tv4.N_PARAMS
MU = float(os.environ.get("PROTY_V5_MU", "1e-5"))


def main():
    t0 = time.time()
    train_shards, y = tv4.build_dataset("train")
    val_shards, vy = tv4.build_dataset("val")
    n_train = len(y)
    print(f"features ready: {n_train} train, {len(vy)} val "
          f"({time.time()-t0:.0f}s)", flush=True)

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

    ref_full = np.load(ROOT / "data" / "theta_v3.npy")
    ref = ref_full[free].copy()

    def val_loss(tp):
        theta = np.zeros(N_PARAMS)
        theta[free] = tp
        return tv4.loss_stream(val_shards, vy, theta)

    CH = 200_000

    def lm_pass(tp, want_gram):
        """Streaming pass over train blocks; returns (raw_train_loss,
        gram, rhs) with the proximal gradient/Hessian folded into
        gram/rhs (sum units, matching train_v4's convention)."""
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
                Xc = tv4.stack_free_rows(Xmg, Xeg, free_mg, free_eg,
                                         lo, hi)
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
        if want_gram:
            # Proximal term in sum units: objective_sum includes
            # (MU*N/2)*mean((tp-ref)^2) = (MU*N/(2*nf))*||tp-ref||^2.
            w = MU * n_train / nf
            gram[np.diag_indices_from(gram)] += w
            rhs += w * (ref - tp)
        return sse / n_train, gram, rhs

    def objective(tp):
        raw, _, _ = lm_pass(tp, False)
        pen = 0.5 * MU * float(np.mean((tp - ref) ** 2))
        return raw, pen

    tp = ref.copy()
    cur_raw, cur_pen = objective(tp)
    curv = val_loss(tp)
    start_raw, start_val = cur_raw, curv
    print(f"start (= shipped params): train {cur_raw:.6f} "
          f"val {curv:.6f} pen 0.000000 mu={MU:g}", flush=True)

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
        cur_obj = cur_raw + cur_pen
        accepted = False
        for s in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = np.clip(tp + s * step, -800, 800)
            cl_raw, cl_pen = objective(cand)
            if cl_raw + cl_pen < cur_obj - 1e-7:
                cv = val_loss(cand)
                print(f"round {rnd}: step {s}: train {cl_raw:.6f} "
                      f"pen {cl_pen:.6f} val {cv:.6f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                tp, cur_raw, cur_pen, curv = cand, cl_raw, cl_pen, cv
                accepted = True
                break
        if not accepted:
            print(f"round {rnd}: no improving step; stop", flush=True)
            break

    theta = np.zeros(N_PARAMS)
    theta[free] = tp
    drift = float(np.sqrt(np.mean((tp - ref) ** 2)))
    print(f"tuned: train {cur_raw:.6f} val {curv:.6f} "
          f"rms drift from shipped {drift:.2f} cp "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(ROOT / "data" / "theta_v5.npy", theta)
    # tv4.emit_inc writes to a hardcoded trained_params_v4.inc path;
    # preserve that artifact, take the emitted text for v5, restore v4.
    v4_inc = ROOT / "src" / "trained_params_v4.inc"
    saved_v4 = v4_inc.read_bytes()
    tv4.emit_inc(theta)
    (ROOT / "src" / "trained_params_v5.inc").write_bytes(
        v4_inc.read_bytes())
    v4_inc.write_bytes(saved_v4)
    report = {
        "method": "warm-start LM from shipped theta_v3 with proximal "
                  "penalty toward it, on the v4 all-position data",
        "mu": MU,
        "train_positions": int(n_train),
        "val_positions": int(len(vy)),
        "loss_start_shipped": {"train": start_raw, "val": start_val},
        "loss_tuned_raw": {"train": cur_raw, "val": curv},
        "rms_drift_cp": drift,
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "data" / "training_report_v5.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
