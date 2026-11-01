#!/usr/bin/env python3
"""SPRT match gate for Proty V2 (adopted for 0.5.0, replacing the fixed
40-game / 55% rule).

Candidate vs base, paired openings: each opening line is played twice
with colours swapped, and the pair is scored pentanomially
(2 / 1.5 / 1 / 0.5 / 0 points from the candidate's perspective). After
every pair the log-likelihood ratio for H0 (candidate <= elo0) vs
H1 (candidate >= elo1) is updated with the normal approximation using
the sample variance of pair scores (the fishtest-style generalised
SPRT); the match stops as soon as the LLR crosses a Wald boundary for
alpha = beta = 0.05, or at the game cap. A floor of MIN_PAIRS pairs is
always played before any stop.

Book is OFF on both engines (the gate measures the engine, not the
book); Threads=1, Hash=256, fixed movetime per move. An engine fault
(crash, no move, illegal move) scores the game as a loss for the
faulty side and both engines are restarted, as in the project's other
match runners.

Usage:
  python3 tests/match_sprt.py CANDIDATE_BIN BASE_BIN [options]
Options:
  --movetime SECONDS   per-move budget (default 0.2)
  --elo1 ELO           H1 advantage (default 15; elo0 is always 0)
  --max-pairs N        game cap in pairs (default 250 = 500 games)
  --min-pairs N        pairs before a stop is allowed (default 10)
  --pgn PATH           PGN output (default sprt.pgn next to this file's
                       run directory, see --outdir)
  --outdir DIR         where pgn/log go (default ./sprt-runs/<stamp>)

Exit code: 0 = H1 accepted (keep), 1 = H0 accepted (reject),
2 = cap reached without a decision.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

MAX_PLIES = 300

OPENINGS = [
    ("Italian Game",     "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
    ("Ruy Lopez",        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"),
    ("French Defense",   "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6"),
    ("Caro-Kann",        "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5"),
    ("Queen's Gambit",   "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7"),
    ("Sicilian Defense", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6"),
    ("London System",    "d2d4 g8f6 c1f4 g7g6 e2e3 f8g7"),
    ("King's Indian",    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6"),
    ("Scotch Game",      "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6"),
    ("English Opening",  "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5"),
]


def expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


class Sprt:
    def __init__(self, elo0: float, elo1: float, alpha: float, beta: float):
        self.mu0 = expected_score(elo0)
        self.mu1 = expected_score(elo1)
        self.lower = math.log(beta / (1.0 - alpha))
        self.upper = math.log((1.0 - beta) / alpha)
        self.pairs = []  # pair scores in [0, 1] from candidate POV

    def add_pair(self, points: float) -> None:
        self.pairs.append(points / 2.0)

    def llr(self) -> float:
        n = len(self.pairs)
        if n < 2:
            return 0.0
        mean = sum(self.pairs) / n
        var = sum((x - mean) ** 2 for x in self.pairs) / n
        if var <= 1e-12:
            # All pairs identical: fall back to a decisive verdict from
            # the mean alone once we have any real sample.
            if mean > self.mu1:
                return self.upper
            if mean < self.mu0:
                return self.lower
            return 0.0
        return n * (self.mu1 - self.mu0) * (mean - (self.mu0 + self.mu1) / 2.0) / var

    def verdict(self, min_pairs: int) -> str:
        if len(self.pairs) < min_pairs:
            return "continue"
        llr = self.llr()
        if llr >= self.upper:
            return "accept_h1"
        if llr <= self.lower:
            return "accept_h0"
        return "continue"

    def score_pct(self) -> float:
        if not self.pairs:
            return 50.0
        return 100.0 * sum(self.pairs) / len(self.pairs)


def start_engine(path: str) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Book": False, "Hash": 256, "Threads": 1})
    return eng


def play_game(cand: chess.engine.SimpleEngine, base: chess.engine.SimpleEngine,
              opening_uci: str, cand_white: bool, movetime: float):
    """Returns (result from candidate POV: 1.0/0.5/0.0, board, fault)."""
    board = chess.Board()
    for u in opening_uci.split():
        board.push_uci(u)
    fault = None
    plies = len(board.move_stack)
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = outcome.result()
            break
        if plies >= MAX_PLIES:
            result = "1/2-1/2"
            break
        cand_turn = (board.turn == chess.WHITE) == cand_white
        engine = cand if cand_turn else base
        try:
            played = engine.play(board, chess.engine.Limit(time=movetime))
            if played.move is None:
                raise chess.engine.EngineError("no move returned")
            board.push(played.move)
        except Exception as e:  # crash / illegal / timeout
            fault = f"{'candidate' if cand_turn else 'base'} fault: {e}"
            # faulty side loses
            if cand_turn:
                result = "0-1" if cand_white else "1-0"
            else:
                result = "1-0" if cand_white else "0-1"
            break
        plies += 1
    if result == "1/2-1/2":
        pts = 0.5
    elif (result == "1-0") == cand_white:
        pts = 1.0
    else:
        pts = 0.0
    return pts, board, fault


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("base")
    ap.add_argument("--movetime", type=float, default=0.2)
    ap.add_argument("--elo1", type=float, default=15.0)
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path("sprt-runs") / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    pgn_path = outdir / "sprt.pgn"
    log_path = outdir / "sprt.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    sprt = Sprt(0.0, args.elo1, 0.05, 0.05)
    log(f"SPRT start: candidate={args.candidate} base={args.base} "
        f"movetime={args.movetime}s elo0=0 elo1={args.elo1} "
        f"bounds=({sprt.lower:.3f},{sprt.upper:.3f}) cap={args.max_pairs} pairs")

    cand = start_engine(args.candidate)
    base = start_engine(args.base)
    verdict = "cap"
    pair_no = 0
    try:
        while pair_no < args.max_pairs:
            name, uci = OPENINGS[pair_no % len(OPENINGS)]
            pair_points = 0.0
            for cand_white in (True, False):
                pts, board, fault = play_game(cand, base, uci, cand_white,
                                              args.movetime)
                pair_points += pts
                game = chess.pgn.Game.from_board(board)
                game.headers["Event"] = "Proty V2 SPRT gate"
                game.headers["White"] = "Candidate" if cand_white else "Base"
                game.headers["Black"] = "Base" if cand_white else "Candidate"
                game.headers["Opening"] = name
                game.headers["Result"] = board.result(claim_draw=True) \
                    if board.outcome(claim_draw=True) else "*"
                if fault:
                    game.headers["Termination"] = fault
                with open(pgn_path, "a") as f:
                    f.write(str(game) + "\n\n")
                if fault:
                    log(f"pair {pair_no + 1}: {fault} — restarting engines")
                    for e in (cand, base):
                        try:
                            e.quit()
                        except Exception:
                            pass
                    cand = start_engine(args.candidate)
                    base = start_engine(args.base)
            sprt.add_pair(pair_points)
            pair_no += 1
            v = sprt.verdict(args.min_pairs)
            log(f"pair {pair_no}: +{pair_points:.1f} pts ({name}) | "
                f"games={2 * pair_no} score={sprt.score_pct():.1f}% "
                f"LLR={sprt.llr():+.3f} -> {v}")
            if v == "accept_h1":
                verdict = "accept_h1"
                break
            if v == "accept_h0":
                verdict = "accept_h0"
                break
    finally:
        for e in (cand, base):
            try:
                e.quit()
            except Exception:
                pass

    games = 2 * len(sprt.pairs)
    log(f"SPRT verdict: {verdict} after {games} games, "
        f"score {sprt.score_pct():.1f}%, LLR {sprt.llr():+.3f}")
    return {"accept_h1": 0, "accept_h0": 1, "cap": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
