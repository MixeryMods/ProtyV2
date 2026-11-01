#!/usr/bin/env python3
"""Merge the per-shard book fragments from extract_v2.py into
data/book.json, applying Anciency's make_book.py recipe: drop
positions whose total weight is below 3 (a single losing appearance),
keep at most 6 moves per position."""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_WEIGHT = 3
MAX_MOVES = 6


def main():
    book = defaultdict(lambda: defaultdict(int))
    n_games = 0
    for path in sorted(glob.glob(str(ROOT / "data" / "positions" / "book-shard*.json"))):
        frag = json.loads(Path(path).read_text())
        for key, moves in frag.items():
            for uci, w in moves.items():
                book[key][uci] += w
    for path in sorted(glob.glob(str(ROOT / "data" / "positions" / "extract-shard*.json"))):
        n_games += json.loads(Path(path).read_text())["book_games"]
    out = {}
    n_moves = 0
    for key, moves in book.items():
        if sum(moves.values()) < MIN_WEIGHT:
            continue
        top = sorted(moves.items(), key=lambda kv: -kv[1])[:MAX_MOVES]
        out[key] = [[uci, w] for uci, w in top]
        n_moves += len(top)
    path = ROOT / "data" / "book.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Book: {len(out)} positions, {n_moves} moves, from "
          f"{n_games} games, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
