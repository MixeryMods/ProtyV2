#!/usr/bin/env python3
"""Featurize the v4 all-position bins (data/positions_v4/shard*.bin)
with the 22 structural features from tools/quiet_features.py, writing
<bin>-ftr.npy next to each bin for tools/train_v4.py."""
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Import quiet_features as a real, named module: process_file fans out to a
# multiprocessing Pool whose workers must resolve its functions by qualified
# name -- an importlib alias ("qf") cannot be pickled into the workers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_features as qf

if __name__ == "__main__":
    import os
    workers = int(os.environ.get("FTR_WORKERS", "3"))
    for bin_path in sorted(glob.glob(str(ROOT / "data" / "positions_v4" / "shard*.bin"))):
        qf.process_file(bin_path, workers)
    print("featurize_v4 done")
