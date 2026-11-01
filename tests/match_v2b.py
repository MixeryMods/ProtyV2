#!/usr/bin/env python3
"""Match harness for Proty V2 strength readings.

Plays paired games (both colours) from a fixed set of 8-ply opening
positions, through python-chess's UCI layer (every move legality-
checked). Book is disabled on both sides: the matches measure the
evaluation + search, not the book.

Usage:
  python3 tests/match_v2.py --a-bin target/release/proty-v2 \
      --b-bin target/release/proty-v2 --b-baseline \
      --games 20 --depth 7
  python3 tests/match_v2.py --a-bin target/release/proty-v2 \
      --b-bin ../anciency/target/release/anciency \
      --games 20 --movetime 200

Score is reported from side A's perspective.
"""
import argparse
import sys

import chess
import chess.engine

OPENINGS_UCI = [
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3",   # Italian
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",        # Scotch
    "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6 g1f3 c7c5",        # London
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",        # King's Indian
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8",        # Nimzo-Indian
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4",        # Slav
    "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4",        # Vienna
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6",        # Queen's Indian
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3",             # Alekhine
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6",        # Scandinavian
]
MAX_PLIES = 400


def opening_board(uci_line):
    board = chess.Board()
    for u in uci_line.split():
        board.push(chess.Move.from_uci(u))
    return board


def play_game(white, black, board, limit):
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        eng = white if board.turn == chess.WHITE else black
        result = eng.play(board, limit)
        assert result.move is not None, "engine returned no move"
        board.push(result.move)
        plies += 1
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-bin", required=True)
    ap.add_argument("--b-bin", required=True)
    ap.add_argument("--a-baseline", action="store_true")
    ap.add_argument("--b-baseline", action="store_true")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime", type=int, default=0,
                    help="milliseconds per move")
    args = ap.parse_args()
    assert bool(args.depth) != bool(args.movetime), "pick depth XOR movetime"
    limit = (chess.engine.Limit(depth=args.depth) if args.depth
             else chess.engine.Limit(time=args.movetime / 1000.0))

    def spawn(path, baseline):
        eng = chess.engine.SimpleEngine.popen_uci(path)
        cfg = {"Book": False, "Hash": 64}
        try:
            eng.configure({**cfg, "Baseline": baseline})
        except Exception:
            eng.configure(cfg)
        return eng

    a = spawn(args.a_bin, args.a_baseline)
    b = spawn(args.b_bin, args.b_baseline)
    score_a = score_b = 0.0
    played = 0
    for opening in OPENINGS_UCI:
        for a_white in (True, False):
            if played >= args.games:
                break
            board = opening_board(opening)
            white, black = (a, b) if a_white else (b, a)
            res = play_game(white, black, board, limit)
            played += 1
            pts_a = (1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5)
            if not a_white:
                pts_a = 1.0 - pts_a
            score_a += pts_a
            score_b += 1.0 - pts_a
            print(f"game {played}: A {'white' if a_white else 'black'} "
                  f"-> {res} (A {score_a} - {score_b} B)", flush=True)
        if played >= args.games:
            break
    a.quit()
    b.quit()
    mode = f"depth {args.depth}" if args.depth else f"{args.movetime} ms"
    print(f"MATCH: A {score_a} - {score_b} B over {played} games ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
