#!/usr/bin/env python3
"""Node-rate bench for Proty V2: fixed-depth searches on three
positions, NPS taken from the engine's own last info line
(nodes / time). Run for the tuned default and with Baseline=true.

Usage: python3 tests/bench_nps.py [engine_path] [depth]
"""
import subprocess
import sys
import time

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "target/release/proty-v2"
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 12

POSITIONS = [
    ("startpos", "position startpos"),
    ("kiwipete", "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
                 "PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("middlegame", "position fen r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/"
                   "PPPQ2PP/R3KB1R w KQ - 3 8"),
]


def bench(baseline):
    proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    def send(s):
        proc.stdin.write(s + "\n")
        proc.stdin.flush()
    send("uci")
    send(f"setoption name Baseline value {'true' if baseline else 'false'}")
    send("setoption name Book value false")
    send("isready")
    total_nodes = total_ms = 0
    for name, pos in POSITIONS:
        send(pos)
        send(f"go depth {DEPTH}")
        last = None
        t0 = time.time()
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("info depth"):
                last = line
            if line.startswith("bestmove"):
                break
        wall = time.time() - t0
        parts = last.split()
        nodes = int(parts[parts.index("nodes") + 1])
        ms = int(parts[parts.index("time") + 1])
        total_nodes += nodes
        total_ms += ms
        print(f"  {name}: {nodes} nodes, {ms} ms engine time "
              f"({nodes / max(ms,1) * 1000:,.0f} nps, wall {wall:.2f}s)")
    send("quit")
    proc.wait(timeout=5)
    print(f"  TOTAL: {total_nodes} nodes in {total_ms} ms = "
          f"{total_nodes / max(total_ms,1) * 1000:,.0f} nps")
    return total_nodes / max(total_ms, 1) * 1000


print(f"bench depth {DEPTH} — tuned params:")
tuned = bench(False)
print(f"bench depth {DEPTH} — baseline params:")
base = bench(True)
print(f"SUMMARY: tuned {tuned:,.0f} nps, baseline {base:,.0f} nps")
