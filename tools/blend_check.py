#!/usr/bin/env python3
"""Blend diagnostic: val/train loss of alpha*v4 + (1-alpha)*v3 parameter
blends, on the cached v4 blocks. If a blend beats BOTH parents on val,
partial adoption of the v4 fit is promising; if none does, the v4 fit's
information is mostly noise for play and the warm-start leash must be
tight."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location(
    "tv4", Path(__file__).resolve().parent / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv4)

val_shards, vy = tv4.build_dataset("val")
v3 = np.load(tv4.ROOT / "data" / "theta_v3.npy")
v4 = np.load(tv4.ROOT / "data" / "theta_v4.npy")

for alpha in (0.0, 0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0):
    theta = (1.0 - alpha) * v3 + alpha * v4
    loss = tv4.loss_stream(val_shards, vy, theta)
    print(f"alpha {alpha:.2f}: val {loss:.6f}", flush=True)
print("BLEND_DONE")
