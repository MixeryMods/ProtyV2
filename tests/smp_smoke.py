#!/usr/bin/env python3
"""Lazy SMP smoke tests for Proty V2 0.4.0.

Checks that multi-threaded search (Threads > 1) starts, terminates,
and obeys stop/clock discipline:
  S1  `uci` lists the Threads option.
  S2  Threads=4, queen-trap position, `go movetime 2000` -> legal
      bestmove within 8 s.
  S3  Threads=2, `go infinite` + stop after ~1 s -> legal bestmove
      within 2 s of the stop.
  S4  Threads=4, `go wtime 30000 btime 30000` from startpos -> legal
      bestmove within 12 s (the clock budget is ~1.5 s).
  S5  Threads=3, knight-rampage position, `go movetime 1000` ->
      legal bestmove within 5 s (stop polling reaches every thread,
      quiescence included).
  S6  Threads=16, `go movetime 500` from startpos -> legal bestmove
      within 8 s (thread spawn storm sanity).

Usage: python3 tests/smp_smoke.py [engine_path]
"""
import queue
import re
import subprocess
import sys
import threading
import time

import chess
import chess.pgn
import io

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "target/release/proty-v2"

P2_SAN = "1. e4 a6 2. Nc3 d6 3. Bb5+ axb5 4. Nxb5 d5 5. exd5 b6 6. Qf3 Qd7 7. Qg4"
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


def legal(board, uci):
    try:
        return chess.Move.from_uci(uci) in board.legal_moves
    except Exception:
        return False


class Session:
    def __init__(self, threads):
        self.proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1)
        self.q = queue.Queue()
        self.uci_lines = []

        def reader():
            for line in self.proc.stdout:
                line = line.strip()
                if line.startswith("option name"):
                    self.uci_lines.append(line)
                self.q.put(line)

        threading.Thread(target=reader, daemon=True).start()
        self.send("uci")
        self.wait_for(lambda l: l == "uciok", 10)
        self.send("setoption name Book value false")
        self.send(f"setoption name Threads value {threads}")
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


START = chess.Board()
P2 = board_after_san(P2_SAN)
HANG = board_after_san(HANG_SAN)

# S1 + S2: Threads option exists; 4-thread movetime search terminates.
s = Session(4)
check("S1 uci lists Threads option",
      any(l.startswith("option name Threads") for l in s.uci_lines),
      f"({len(s.uci_lines)} options seen)")
bm, dt = s.go(P2, "go movetime 2000", timeout=10)
print(f"  S2 Threads=4 P2 movetime 2000 -> {bm} in {dt:.2f}s")
check("S2 Threads=4 movetime search returns legal move within 8 s",
      bm is not None and dt < 8.0 and legal(P2, bm), f"(got {bm})")
s.close()

# S3: infinite search + stop with 2 threads.
s = Session(2)
s.send("position startpos")
s.send("go infinite")
time.sleep(1.2)
t0 = time.time()
s.send("stop")
line = s.wait_for(lambda l: l.startswith("bestmove"), 4)
dt = time.time() - t0
bm = line.split()[1] if line else None
print(f"  S3 Threads=2 infinite+stop -> {bm} in {dt:.2f}s after stop")
check("S3 stop during threaded search answers within 2 s",
      bm is not None and dt < 2.0 and legal(START, bm), f"(got {bm})")
s.close()

# S4: clock-budgeted search with 4 threads respects the budget.
s = Session(4)
bm, dt = s.go(START, "go wtime 30000 btime 30000", timeout=14)
print(f"  S4 Threads=4 wtime 30000 -> {bm} in {dt:.2f}s")
check("S4 threaded clock search returns legal move within 12 s",
      bm is not None and dt < 12.0 and legal(START, bm), f"(got {bm})")
s.close()

# S5: rampage position, 3 threads, hard 1 s budget.
s = Session(3)
bm, dt = s.go(HANG, "go movetime 1000", timeout=7)
print(f"  S5 Threads=3 HANG movetime 1000 -> {bm} in {dt:.2f}s")
check("S5 threaded search on rampage position terminates within 5 s",
      bm is not None and dt < 5.0 and legal(HANG, bm), f"(got {bm})")
s.close()

# S6: max threads spawn storm.
s = Session(16)
bm, dt = s.go(START, "go movetime 500", timeout=10)
print(f"  S6 Threads=16 movetime 500 -> {bm} in {dt:.2f}s")
check("S6 Threads=16 search returns legal move within 8 s",
      bm is not None and dt < 8.0 and legal(START, bm), f"(got {bm})")
s.close()

print("ALL OK" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
