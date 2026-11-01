#!/usr/bin/env python3
"""Stage-by-stage RSS probe of train_v4.build_all on shard0-train, to find
which stage OOM-kills this VM."""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location(
    "tv4", Path(__file__).resolve().parent / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv4)


def rss():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return line.strip()
    return '?'


print('start', rss(), flush=True)
rec = tv4.tv2.load_records('data/positions_v4/shard0-train.bin')
print('rec loaded', len(rec), rss(), flush=True)
ftr = np.concatenate([np.load(f) for f in sorted(glob.glob(
    str(tv4.ROOT / 'data/positions_v4/shard0-train-ftr.npy')))])
print('ftr loaded', ftr.shape, ftr.dtype, rss(), flush=True)
Xmg, Xeg, y = tv4.tv2.build_features(rec)
print('build_features done', Xmg.shape, Xmg.nnz, rss(), flush=True)
phase = tv4.compute_phase(rec)
print('phase done', rss(), flush=True)
w_mg = (phase / 24.0).astype(np.float32)
w_eg = (1.0 - phase / 24.0).astype(np.float32)
f32 = ftr.astype(np.float32, copy=False)
Emg = tv4.extra_csr(f32, w_mg)
print('Emg done', Emg.nnz, rss(), flush=True)
Eeg = tv4.extra_csr(f32, w_eg)
print('Eeg done', Eeg.nnz, rss(), flush=True)
from scipy.sparse import hstack
Fmg = hstack([Xmg, Emg], format='csr')
print('hstack mg done', rss(), flush=True)
Feg = hstack([Xeg, Eeg], format='csr')
print('ALL DONE', Fmg.shape, Feg.shape, rss(), flush=True)
