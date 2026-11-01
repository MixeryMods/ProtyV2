#!/usr/bin/env python3
"""Build the v4 trainer's CSR cache for one shard/split, in ~1M-row
blocks (feat-shard{s}-{split}-b{k}-{mg,eg}.npz + -y.npy).

Why blocks: tv2.build_features accumulates per-chunk CSR parts and
vstacks them; on a full 4.56M-row shard that peaks over 5 GB and
OOM-kills this VM (exit 137, three times on 2026-11-01). A 1M-row
block peaks around 2.5 GB. Finished blocks persist in
data/positions_v4/, so a killed run resumes at the next block, and
train_v4.build_dataset streams the block files as its shard list.

Usage: build_cache_v4.py <shard 0|1> <train|val>
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

import numpy as np
from scipy.sparse import hstack, save_npz

spec = importlib.util.spec_from_file_location(
    "tv4", Path(__file__).resolve().parent / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv4)

BLOCK = 1_000_000


def build_block(rec, ftr):
    """One row block -> (Xmg, Xeg, y): the per-row math of
    train_v4.build_all, restricted to these rows."""
    Xmg, Xeg, y = tv4.tv2.build_features(rec)
    f32 = ftr.astype(np.float32, copy=False)
    phase = tv4.compute_phase(rec)
    w_mg = (phase / 24.0).astype(np.float32)
    w_eg = (1.0 - phase / 24.0).astype(np.float32)
    Emg = tv4.extra_csr(f32, w_mg)
    Eeg = tv4.extra_csr(f32, w_eg)
    return (hstack([Xmg, Emg], format="csr"),
            hstack([Xeg, Eeg], format="csr"), y)


def main():
    shard, split = int(sys.argv[1]), sys.argv[2]
    d = tv4.FEAT_DIR
    rec = tv4.tv2.load_records(f"data/positions_v4/shard{shard}-{split}.bin")
    ftr = np.concatenate([np.load(f) for f in sorted(glob.glob(
        str(d / f"shard{shard}-{split}-ftr.npy")))])
    assert len(ftr) == len(rec), (len(ftr), len(rec))
    nblocks = (len(rec) + BLOCK - 1) // BLOCK
    print(f"shard{shard}-{split}: {len(rec)} rows, {nblocks} blocks",
          flush=True)
    for k in range(nblocks):
        mg_p = d / f"feat-shard{shard}-{split}-b{k}-mg.npz"
        eg_p = d / f"feat-shard{shard}-{split}-b{k}-eg.npz"
        y_p = d / f"feat-shard{shard}-{split}-b{k}-y.npy"
        if mg_p.exists() and eg_p.exists() and y_p.exists():
            print(f"  block {k}: cached, skip", flush=True)
            continue
        lo, hi = k * BLOCK, min((k + 1) * BLOCK, len(rec))
        Xmg, Xeg, y = build_block(rec[lo:hi], ftr[lo:hi])
        save_npz(mg_p, Xmg)
        save_npz(eg_p, Xeg)
        np.save(y_p, y)
        del Xmg, Xeg, y
        print(f"  block {k}: saved rows {lo}:{hi}", flush=True)
    print(f"shard{shard}-{split}: cache complete", flush=True)


if __name__ == "__main__":
    main()
