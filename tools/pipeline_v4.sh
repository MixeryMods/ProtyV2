#!/bin/bash
# Chain the rest of the 0.7.5 pipeline once the shard0-train cache build
# (already running, started by hand) exits: build the remaining caches,
# verify every block file exists, then run the v4 fit. Any missing block
# aborts before train_v4, because train_v4 would otherwise fall back to
# the full-shard build_all that OOM-kills this VM.
set -u
cd /home/hatch/workspace/proty-v2
D=data/positions_v4

while pgrep -f 'build_cache_v4.py 0 train' > /dev/null; do sleep 15; done
echo "[pipeline] shard0-train builder exited $(date +%T)"

python3 -u tools/build_cache_v4.py 1 train >> $D/cache-rest.log 2>&1
python3 -u tools/build_cache_v4.py 0 val   >> $D/cache-rest.log 2>&1
python3 -u tools/build_cache_v4.py 1 val   >> $D/cache-rest.log 2>&1
echo "[pipeline] cache builds done $(date +%T)"

ok=1
for s in 0 1; do
  for k in 0 1 2 3 4; do
    for suf in mg.npz eg.npz y.npy; do
      f=$D/feat-shard$s-train-b$k-$suf
      [ -f "$f" ] || { echo "[pipeline] MISSING $f"; ok=0; }
    done
  done
  for suf in mg.npz eg.npz y.npy; do
    f=$D/feat-shard$s-val-b0-$suf
    [ -f "$f" ] || { echo "[pipeline] MISSING $f"; ok=0; }
  done
done
[ $ok -eq 1 ] || { echo "[pipeline] ABORT: cache incomplete"; exit 1; }

echo "[pipeline] cache verified, starting train_v4 $(date +%T)"
python3 -u tools/train_v4.py > $D/train.log 2>&1
echo "[pipeline] train_v4 exited rc=$? $(date +%T)"
