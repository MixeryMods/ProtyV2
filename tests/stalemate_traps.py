#!/usr/bin/env python3
"""Stalemate-trap suite for Proty V2 (ported from Proty V1's suite).

Each position: the side to move is winning (K+Q vs K material), a
tempting legal move stalemates the opponent immediately, and winning
alternatives exist. The engine passes a position iff its bestmove is
NOT the stalemating move and does not leave the opponent stalemated.

Run: python3 tests/stalemate_traps.py [engine_path] [depth]
"""
import sys
import chess
import chess.engine

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "target/release/proty-v2"
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 12

TRAPS = [
    ("T1 queen-b6 trap (Ka8)", "k7/8/8/2Q5/8/8/8/K7 w - - 0 1", "c5b6"),
    ("T2 queen-g6 trap (Kh8)", "7k/8/8/5Q2/8/8/8/K7 w - - 0 1", "f5g6"),
    ("T3 queen-b6 trap + pawn", "k7/8/8/2Q5/8/8/5P2/K7 w - - 0 1", "c5b6"),
    ("T4 queen-g6 trap + bishop (mate in 1 exists)",
     "7k/8/8/5Q2/8/8/2B5/K7 w - - 0 1", "f5g6"),
    ("T5 two queens (mate in 1 exists)", "k7/8/8/2Q5/8/8/8/KQ6 w - - 0 1", "c5b6"),
    ("T6 black queen-b3 trap (Ka1)", "7k/8/8/8/2q5/8/8/K7 b - - 0 1", "c4b3"),
]

def main():
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Book": False})
    passed = 0
    for name, fen, stale_uci in TRAPS:
        board = chess.Board(fen)
        stale = chess.Move.from_uci(stale_uci)
        assert stale in board.legal_moves
        probe = board.copy(); probe.push(stale)
        assert probe.is_stalemate(), f"{name}: trap move does not stalemate?!"
        result = engine.play(board, chess.engine.Limit(depth=DEPTH))
        bm = result.move
        after = board.copy(); after.push(bm)
        ok = bm != stale and not after.is_stalemate()
        passed += ok
        print(f"{name}: bestmove {board.uci(bm)} -> {'PASS' if ok else 'FAIL'}")
    engine.quit()
    print(f"{passed}/{len(TRAPS)} traps passed")
    sys.exit(0 if passed == len(TRAPS) else 1)

if __name__ == "__main__":
    main()
