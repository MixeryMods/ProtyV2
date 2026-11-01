#!/usr/bin/env python3
"""Gauss-Newton (LM-style) polish for Proty V2's tuned parameters.

Starts from data/theta_trained.npy (the ridge-LS solution emitted by
tools/train_v2.py) and runs monotone Gauss-Newton rounds on the TRUE
Texel loss mean((sigmoid(eval/SCALE) - y)^2) — the refinement stage
of Anciency's tuner (their LM rounds carried their fit most of the
way). Each round: build the weighted normal equations with sparse
chunked grams, solve with the midgame/endgame split penalty, then
line-search the step on the true loss and accept only if it improves.
Re-emits src/trained_params.inc and updates data/training_report.json
(loss_tuned / loss_lm) only when the polished parameters beat the
starting ones on the training loss.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

import importlib.util
spec = importlib.util.spec_from_file_location(
    "tv2", Path(__file__).resolve().parent / "train_v2.py")
tv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv2)

ROOT = tv2.ROOT
SCALE = tv2.SCALE
BLOCK, NV, NPST = tv2.BLOCK, tv2.NV, tv2.NPST
N_PARAMS = tv2.N_PARAMS


def main(rounds=6):
    t0 = time.time()
    rec_train = tv2.load_records("data/positions/shard*-train.bin")
    rec_val = tv2.load_records("data/positions/shard*-val.bin")
    Xmg, Xeg, y = tv2.build_features(rec_train)
    Vmg, Veg, vy = tv2.build_features(rec_val)
    print(f"features rebuilt ({time.time()-t0:.0f}s)", flush=True)

    free = np.array([c for c in range(N_PARAMS)
                     if not (0 <= c < NV or BLOCK <= c < BLOCK + NV)])
    free_mg = np.array([c for c in free if c < BLOCK])
    free_eg = np.array([c - BLOCK for c in free if c >= BLOCK])
    Xf = hstack([Xmg[:, free_mg], Xeg[:, free_eg]], format="csr")
    Vf = hstack([Vmg[:, free_mg], Veg[:, free_eg]], format="csr")
    del Xmg, Xeg, Vmg, Veg
    nf = len(free)
    pos_of = {c: i for i, c in enumerate(free)}
    pairs = [(pos_of[NV + k], pos_of[BLOCK + NV + k]) for k in range(NPST + 1)]
    pi = np.array([a for a, _ in pairs])
    pj = np.array([b for _, b in pairs])

    theta = np.load(ROOT / "data" / "theta_trained.npy")
    tp = theta[free].copy()

    def losses(tp_):
        e = Xf @ tp_
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        tl = float(np.mean((p - y) ** 2))
        ev = Vf @ tp_
        pv = 1.0 / (1.0 + np.exp(-ev / SCALE))
        vl = float(np.mean((pv - vy) ** 2))
        return tl, vl

    cur, curv = losses(tp)
    print(f"start: train {cur:.6f} val {curv:.6f}", flush=True)
    start = (cur, curv)
    CH = 300_000
    for rnd in range(1, rounds + 1):
        e = Xf @ tp
        p = 1.0 / (1.0 + np.exp(-e / SCALE))
        r = y - p
        dp = p * (1.0 - p) / SCALE          # sigmoid derivative wrt e
        w = dp * dp                          # Gauss-Newton weights
        gram = np.zeros((nf, nf))
        rhs = np.zeros(nf)
        for lo in range(0, len(y), CH):
            hi = min(lo + CH, len(y))
            Xc = Xf[lo:hi]
            Xs = Xc.multiply(np.sqrt(w[lo:hi])[:, None]).tocsr()
            gram += (Xs.T @ Xs).toarray()
            rhs += Xc.T @ (dp[lo:hi] * r[lo:hi])
            del Xc, Xs
        md = float(np.mean(np.diag(gram)))
        lam = 0.02 * md
        D = np.zeros_like(gram)
        D[pi, pi] += lam
        D[pj, pj] += lam
        D[pi, pj] -= lam
        D[pj, pi] -= lam
        split_vec = np.zeros(nf)
        split_vec[pi] = tp[pi] - tp[pj]
        split_vec[pj] = -(tp[pi] - tp[pj])
        # Penalty gradient: D @ tp == lam * split_vec by construction.
        rhs = rhs - lam * split_vec
        delta = np.linalg.solve(gram + D + (1e-4 * md) * np.eye(nf), rhs)
        accepted = False
        for s in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = np.clip(tp + s * delta, -800, 800)
            cl, cv = losses(cand)
            if cl < cur - 1e-7:
                print(f"round {rnd}: step {s}: train {cl:.6f} val "
                      f"{cv:.6f} ({time.time()-t0:.0f}s)", flush=True)
                tp, cur, curv = cand, cl, cv
                accepted = True
                break
        if not accepted:
            print(f"round {rnd}: no improving step; stop", flush=True)
            break

    if cur < start[0] - 1e-7:
        theta[free] = tp
        np.save(ROOT / "data" / "theta_trained.npy", theta)
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

        inc = f"""EvalParams {{
    value_mg: {fmt_arr(mg[0:6])},
    value_eg: {fmt_arr(eg[0:6])},
    pst_mg: {fmt_pst(mg[6:6 + 384])},
    pst_eg: {fmt_pst(eg[6:6 + 384])},
    bishop_pair_mg: {int(round(mg[NV + NPST]))},
    bishop_pair_eg: {int(round(eg[NV + NPST]))},
}}"""
        (ROOT / "src" / "trained_params.inc").write_text(inc + "\n")
        report = json.loads(
            (ROOT / "data" / "training_report.json").read_text())
        report["loss_lm_polish"] = {"train": cur, "val": curv}
        report["loss_tuned"] = {"train": cur, "val": curv}
        names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        report["implied_value_mg_from_pst_mean"] = {
            names[i]: int(round(float(np.mean(mg[6 + i * 64:6 + (i + 1) * 64]))))
            for i in range(6)}
        report["implied_value_eg_from_pst_mean"] = {
            names[i]: int(round(float(np.mean(eg[6 + i * 64:6 + (i + 1) * 64]))))
            for i in range(6)}
        report["bishop_pair_mg"] = int(round(mg[NV + NPST]))
        report["bishop_pair_eg"] = int(round(eg[NV + NPST]))
        report["lm_rounds_seconds"] = round(time.time() - t0, 1)
        (ROOT / "data" / "training_report.json").write_text(
            json.dumps(report, indent=2))
        print(f"POLISHED: train {cur:.6f} val {curv:.6f} — emitted")
    else:
        print(f"no improvement over start {start}; parameters unchanged")
    print(json.dumps({"start": start, "final": (cur, curv)}, indent=2))


if __name__ == "__main__":
    main()
