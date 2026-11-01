#!/bin/bash
# Chain for the warm-start (v5) candidate: wait for train_v5 to exit,
# build the candidate tree with trained_params_v5.inc, UCI smoke,
# cargo tests, the 7-position sanity check, then the 200-game SPRT
# gate vs frozen 0.7.0. Aborts loudly if the tuner produced nothing.
set -u
SRC=/home/hatch/workspace/proty-v2
CAND=/home/hatch/workspace/proty-v2-candidate-v5
export PATH="$HOME/.cargo/bin:$PATH"

while pgrep -f 'train_v5[.]py' > /dev/null; do sleep 15; done
echo "[pipeline-v5] tuner exited $(date +%T)"

if [ ! -f $SRC/data/theta_v5.npy ] || [ ! -f $SRC/src/trained_params_v5.inc ]; then
  echo "[pipeline-v5] ABORT: theta_v5 / trained_params_v5.inc missing"
  exit 1
fi

rm -rf $CAND
mkdir -p $CAND
cd $SRC && cp -r src Cargo.toml Cargo.lock $CAND/
mkdir -p $CAND/data && cp data/book.json $CAND/data/
cp $SRC/src/trained_params_v5.inc $CAND/src/trained_params.inc

cd $CAND
cargo build --release > build.log 2>&1
echo "[pipeline-v5] build rc=$? $(date +%T)"
[ -x target/release/proty-v2 ] || { echo "[pipeline-v5] ABORT: no binary"; exit 1; }

printf 'uci\nisready\nposition startpos\ngo depth 8\nquit\n' | timeout 30 ./target/release/proty-v2 > uci-smoke.log 2>&1
grep -q 'uciok' uci-smoke.log && echo "[pipeline-v5] uci smoke ok" || echo "[pipeline-v5] UCI SMOKE FAILED"

cargo test > cargo-test.log 2>&1
echo "[pipeline-v5] cargo test rc=$? $(date +%T)"

python3 /home/hatch/workspace/chess/proty-games/ab_test_v5.py > /home/hatch/workspace/chess/proty-games/ab-test-v5.log 2>&1
echo "[pipeline-v5] sanity done $(date +%T)"

cd $SRC
python3 tests/match_sprt.py $CAND/target/release/proty-v2 gate-opponents/proty-v2-0.7.0 --max-pairs 100 --outdir sprt-runs/v5-warm-vs-070 > sprt-runs-v5.out 2>&1
echo "[pipeline-v5] SPRT exited rc=$? $(date +%T)"
echo "PIPELINE_V5_DONE"
