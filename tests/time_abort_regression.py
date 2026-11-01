#!/usr/bin/env python3
"""Regression tests for the search clock defects — ported from Proty
V1's suite (where these defects shipped and ruined every timed game)
and adapted to Proty V2:

  Defect 1 — partial-iteration results: a clock-aborted iteration must
      be discarded whole; the returned move comes from the last
      COMPLETED iteration (V1 played generation-order garbage).
  Defect 2 — runaway searches: clock/stop must be polled in every
      recursive path (quiescence included) and a hard node failsafe
      must bound every time-budgeted search.

Tests:
  P2 (queen trap): `go movetime 3000` and `go depth 18 movetime 3000`
      must return d7g4 (Qxg4 wins the queen).
  G1: `go movetime 3000` must NOT return c5c4 (V1's poisoned push).
  HANG (knight-rampage position): `go movetime 1000` must return a
      legal bestmove within 5 s; `stop` mid-search must be answered
      with a legal bestmove within 2 s.

Positions are built by parsing the SAN game lines with python-chess,
so the move lists are legal by construction. Book is disabled so a
real search must produce every move.

Usage: python3 tests/time_abort_regression.py [engine_path]
"""
import io
import queue
import re
import subprocess
import sys
import threading
import time

import chess
import chess.pgn

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "target/release/proty-v2"

P2_SAN = "1. e4 a6 2. Nc3 d6 3. Bb5+ axb5 4. Nxb5 d5 5. exd5 b6 6. Qf3 Qd7 7. Qg4"
G1_SAN = "1. e4 c6 2. Nf3 d6 3. Nc3 c5 4. b3"
HANG_SAN = ("1. a3 d5 2. b3 Nf6 3. c3 e5 4. d3 Ng4 5. h3 Nxf2 6. Nf3 Nxd1 "
            "7. e3 Nf2 8. c4 Nxh1 9. b4 Nf2 10. cxd5 Nxd3+ 11. Kd1 Nxc1 "
            "12. g3 Qxd5+ 13. Kxc1 Bxb4")

fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name +
          (" " + extra if extra else ""), flush=True)
    if not cond:
        fails.append(name)


def board_after_san(san_line):
    board = chess.Board()
    for tok in san_line.split():
        if re.fullmatch(r"\d+\.(\.\.)?", tok):
            continue
        board.push_san(tok)
    return board


def uci_line(board):
    return " ".join(m.uci() for m in board.move_stack)


class Session:
    def __init__(self):
        self.proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1)
        self.q = queue.Queue()
        self.last_info = None

        def reader():
            for line in self.proc.stdout:
                line = line.strip()
                if line.startswith("info depth"):
                    self.last_info = line
                self.q.put(line)

        threading.Thread(target=reader, daemon=True).start()
        self.send("uci")
        self.send("setoption name Book value false")
        self.send("isready")
        self.wait_for(lambda l: l == "readyok", 10)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_for(self, pred, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                break
            if pred(line):
                return line
        return None

    def go(self, board, go_cmd, timeout):
        self.last_info = None
        self.send("position startpos moves " + uci_line(board))
        t0 = time.time()
        self.send(go_cmd)
        line = self.wait_for(lambda l: l.startswith("bestmove"), timeout)
        dt = time.time() - t0
        if line is None:
            return None, dt
        return line.split()[1], dt

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def legal(board, uci):
    try:
        return chess.Move.from_uci(uci) in board.legal_moves
    except Exception:
        return False


P2 = board_after_san(P2_SAN)
G1 = board_after_san(G1_SAN)
HANG = board_after_san(HANG_SAN)

s = Session()

bm, dt = s.go(P2, "go movetime 3000", timeout=15)
print(f"  P2 movetime 3000 -> {bm} in {dt:.2f}s | {s.last_info}")
check("P2 go movetime 3000 returns d7g4", bm == "d7g4", f"(got {bm})")

bm, dt = s.go(P2, "go depth 18 movetime 3000", timeout=15)
print(f"  P2 depth 18 movetime 3000 -> {bm} in {dt:.2f}s | {s.last_info}")
check("P2 go depth 18 movetime 3000 returns d7g4", bm == "d7g4",
      f"(got {bm})")

bm, dt = s.go(G1, "go movetime 3000", timeout=15)
print(f"  G1 movetime 3000 -> {bm} in {dt:.2f}s | {s.last_info}")
check("G1 go movetime 3000 does not return c5c4",
      bm is not None and bm != "c5c4" and legal(G1, bm), f"(got {bm})")

bm, dt = s.go(HANG, "go movetime 1000", timeout=8)
print(f"  HANG movetime 1000 -> {bm} in {dt:.2f}s | {s.last_info}")
check("HANG go movetime 1000 returns legal bestmove within 5 s",
      bm is not None and dt < 5.0 and legal(HANG, bm),
      f"(got {bm} in {dt:.2f}s)")

s.send("position startpos moves " + uci_line(HANG))
s.send("go movetime 60000")
time.sleep(1.5)
t0 = time.time()
s.send("stop")
line = s.wait_for(lambda l: l.startswith("bestmove"), 4)
dt = time.time() - t0
bm = line.split()[1] if line else None
print(f"  HANG stop mid-search -> {bm} in {dt:.2f}s after stop | {s.last_info}")
check("stop mid-search produces legal bestmove within 2 s",
      bm is not None and dt < 2.0 and legal(HANG, bm),
      f"(got {bm} in {dt:.2f}s)")

s.close()
print("ALL OK" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
