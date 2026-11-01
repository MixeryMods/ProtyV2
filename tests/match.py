#!/usr/bin/env python3
"""Self-play and trained-vs-baseline matches for Anciency, driven through
python-chess's UCI layer (which validates every move's legality).

Usage:
  python3 tests/match.py selfplay          one full game, engine vs itself
  python3 tests/match.py match [games]     trained vs baseline, book off,
                                           corpus openings, colors swapped
"""
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / "target" / "release" / "anciency")
MOVETIME = 0.06
MAX_PLIES = 300


def play_game(white: chess.engine.SimpleEngine, black: chess.engine.SimpleEngine,
              board: chess.Board) -> str:
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        engine = white if board.turn == chess.WHITE else black
        result = engine.play(board, chess.engine.Limit(time=MOVETIME))
        assert result.move is not None, "engine returned no move"
        board.push(result.move)  # python-chess raises on illegal moves
        plies += 1
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    if board.is_game_over(claim_draw=True):
        return "1/2-1/2"
    return "1/2-1/2 (ply cap)"


def configure(baseline: bool) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(BIN)
    eng.configure({"Book": False, "Baseline": baseline, "Hash": 16})
    return eng


def selfplay() -> int:
    eng = configure(baseline=False)
    board = chess.Board()
    result = play_game(eng, eng, board)
    print(f"Self-play game: {board.ply()} plies, result {result}")
    print("Final FEN:", board.fen())
    moves = " ".join(m.uci() for m in board.move_stack)
    print("Moves (UCI):", moves)
    eng.quit()
    return 0


def corpus_openings(n: int):
    """n distinct 8-ply opening positions from the merged corpus."""
    fens, seen = [], set()
    with open(ROOT / "data" / "ancient_corpus.pgn") as fh:
        while len(fens) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < 8:
                continue
            for m in moves[:8]:
                board.push(m)
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                continue
            seen.add(key)
            fens.append(board.fen())
    return fens


def match(n_games: int) -> int:
    openings = corpus_openings((n_games + 1) // 2)
    trained = configure(baseline=False)
    baseline = configure(baseline=True)
    score_t = score_b = 0.0
    results = []
    game_no = 0
    for fen in openings:
        for trained_white in (True, False):
            if game_no >= n_games:
                break
            board = chess.Board(fen)
            white, black = (trained, baseline) if trained_white else (baseline, trained)
            res = play_game(white, black, board)
            game_no += 1
            if res.startswith("1-0"):
                pts_t = 1.0 if trained_white else 0.0
            elif res.startswith("0-1"):
                pts_t = 0.0 if trained_white else 1.0
            else:
                pts_t = 0.5
            score_t += pts_t
            score_b += 1.0 - pts_t
            results.append(f"game {game_no}: {'T' if trained_white else 'B'} white -> {res}")
            print(results[-1], f"(trained {score_t} - {score_b} baseline)", flush=True)
    trained.quit()
    baseline.quit()
    print(f"MATCH RESULT: trained {score_t} - {score_b} baseline over {game_no} games")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selfplay"
    if mode == "selfplay":
        sys.exit(selfplay())
    else:
        sys.exit(match(int(sys.argv[2]) if len(sys.argv) > 2 else 30))
