#!/usr/bin/env python3
"""Match harness for the Proty V2 0.4.0 Lazy SMP gate.

Same protocol as tests/match_v2.py (paired colours from the same 10
opening positions, book off both sides, score from side A's
perspective), plus per-side Threads options. Engines that do not
know the Threads option (e.g. the 0.3.0 binary) are configured
without it via the fallback.

Usage:
  python3 tests/match_smp.py --a-bin target/release/proty-v2 \
      --b-bin release-0.3.0-backup/target/release/proty-v2 \
      --a-threads 2 --games 40 --movetime 200
"""
import argparse
import sys

import chess
import chess.engine

OPENINGS_UCI = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",   # Ruy Lopez
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",   # Sicilian
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",             # QGD
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",             # French
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",             # Gruenfeld
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5",   # Caro-Kann
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",             # English
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",             # Dutch
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",             # Pirc
    "g1f3 d7d5 g2g3 c7c6 f1g2 c8g4",             # Reti
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
    ap.add_argument("--a-threads", type=int, default=1)
    ap.add_argument("--b-threads", type=int, default=1)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=int, default=200,
                    help="milliseconds per move")
    args = ap.parse_args()
    limit = chess.engine.Limit(time=args.movetime / 1000.0)

    def spawn(path, threads):
        eng = chess.engine.SimpleEngine.popen_uci(path)
        cfg = {"Book": False, "Hash": 64}
        try:
            eng.configure({**cfg, "Threads": threads})
        except Exception:
            eng.configure(cfg)
        return eng

    a = spawn(args.a_bin, args.a_threads)
    b = spawn(args.b_bin, args.b_threads)
    score_a = score_b = 0.0
    played = 0
    while played < args.games:
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
    print(f"MATCH: A {score_a} - {score_b} B over {played} games "
          f"({args.movetime} ms, A threads {args.a_threads}, "
          f"B threads {args.b_threads})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
