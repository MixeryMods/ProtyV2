#!/usr/bin/env python3
"""Tablebase smoke tests for Proty V2 0.7.0.

Requires a real Syzygy 3-4-5-man set. Point PROTY_SYZYGY_PATH at
the directory holding the .rtbw/.rtbz files (default:
~/workspace/syzygy-5man/tables). If the directory is missing the
suite prints SKIP and exits 0 — the no-tables behaviour is covered
by the other suites; this one exists to prove the engine actually
loads tables, probes them with the right signs, and converts a
won ending by DTZ play.
"""
import os
import subprocess
import sys

import chess

ENG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "target", "release", "proty-v2")
TB = os.environ.get(
    "PROTY_SYZYGY_PATH",
    os.path.expanduser("~/workspace/syzygy-5man/tables"))

ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok = ok and cond


def session(cmds, timeout=30):
    p = subprocess.run([ENG], input="\n".join(cmds) + "\n",
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout.splitlines()


def probe(fen, go="go movetime 2000"):
    lines = session([f"setoption name SyzygyPath value {TB}",
                     "setoption name Book value false",
                     f"position fen {fen}", go, "quit"])
    infos = [l for l in lines if l.startswith("info depth")]
    bm = [l for l in lines if l.startswith("bestmove")]
    return (infos[-1] if infos else "", bm[-1].split()[1] if bm else "")


def score_of(info):
    parts = info.split()
    return int(parts[parts.index("score") + 2])


def main():
    if not os.path.isdir(TB):
        print(f"SKIP: no tablebase directory at {TB}")
        return 0

    lines = session(["uci", "quit"])
    check("SyzygyPath option listed",
          any("option name SyzygyPath" in l for l in lines))

    info, _ = probe("8/8/8/4k3/8/8/3QK3/8 w - - 0 1")
    check("KQvK white is a TB win", "depth 0" in info and score_of(info) > 50_000, info)
    info, _ = probe("8/8/8/4k3/8/8/3QK3/8 b - - 0 1")
    check("KQvK black is a TB loss", "depth 0" in info and score_of(info) < -50_000, info)
    info, _ = probe("8/8/8/4k3/3p4/8/3RK3/8 w - - 0 1")
    check("KRvKP white is a TB win", "depth 0" in info and score_of(info) > 50_000, info)
    info, _ = probe("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    check("KvK is a TB draw", "depth 0" in info and score_of(info) == 0, info)
    info, _ = probe("8/8/8/k7/8/P7/1K6/2B5 w - - 0 1")
    check("wrong-colour KBP is a TB draw",
          "depth 0" in info and score_of(info) == 0, info)

    # Inner-search WDL probe: 6-piece root, Qxd4 enters a won
    # 5-man; the score must be the ply-adjusted TB win, far beyond
    # any heuristic value.
    info, bm = probe("8/8/8/4k3/3p4/8/3QK1PP/8 w - - 0 1", go="go depth 3")
    check("6-piece root probes TB in search",
          score_of(info) > 50_000 and bm == "d2d4", f"{info} bm={bm}")

    # DTZ conversion: TB side mates KQvK against a searching
    # defender, quickly and well inside the fifty-move budget.
    board = chess.Board("8/8/8/4k3/8/8/3QK3/8 w - - 0 1")
    plies = 0
    while not board.is_game_over() and plies < 60:
        if board.turn == chess.WHITE:
            _, mv = probe(board.fen(), go="go movetime 500")
        else:
            lines = session(["setoption name Book value false",
                             f"position fen {board.fen()}",
                             "go depth 3", "quit"])
            bm = [l for l in lines if l.startswith("bestmove")]
            mv = bm[-1].split()[1]
        board.push_uci(mv)
        plies += 1
    check("KQvK converted to mate", board.is_checkmate(), f"{plies} plies")

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
