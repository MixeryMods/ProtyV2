#!/usr/bin/env python3
"""Lazy SMP scaling bench for Proty V2 0.4.0: wall-clock time to a
fixed depth at Threads=1 vs Threads=2 (this VM has 2 cores), on the
start position and Kiwipete. Book off, Hash 64. Each configuration
runs twice; the best wall time is reported (the machine is otherwise
unloaded). The engine's own last info line gives the main thread's
nodes/time for reference — with helpers sharing the TT, the main
thread's tree is not directly comparable across thread counts, so
wall time to depth is the honest metric.

Usage: python3 tests/bench_threads.py [engine_path]
"""
import subprocess
import sys
import time

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "target/release/proty-v2"

CASES = [
    ("startpos", "position startpos", 11),
    ("kiwipete",
     "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
     "PPPBBPPP/R3K2R w KQkq - 0 1", 10),
]


def run_once(threads, pos_cmd, depth):
    proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(s):
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    send("uci")
    send("setoption name Book value false")
    send("setoption name Hash value 64")
    send(f"setoption name Threads value {threads}")
    send("isready")
    send(pos_cmd)
    t0 = time.time()
    send(f"go depth {depth}")
    last = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("info depth"):
            last = line
        if line.startswith("bestmove"):
            break
    wall = time.time() - t0
    send("quit")
    proc.wait(timeout=5)
    parts = last.split()
    nodes = int(parts[parts.index("nodes") + 1])
    ms = int(parts[parts.index("time") + 1])
    return wall, nodes, ms


for name, pos_cmd, depth in CASES:
    results = {}
    for threads in (1, 2):
        runs = [run_once(threads, pos_cmd, depth) for _ in range(2)]
        best = min(runs, key=lambda r: r[0])
        results[threads] = best
        print(f"{name} depth {depth} Threads={threads}: "
              f"wall {best[0]:.2f}s (main nodes {best[1]:,}, "
              f"engine time {best[2]} ms)")
    speedup = results[1][0] / results[2][0]
    print(f"  -> Threads=2 speedup: {speedup:.2f}x")
