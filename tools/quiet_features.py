#!/usr/bin/env python3
"""Extra evaluation features for Proty V2's Phase B (2.1.0) retune.

Computes, for every 66-byte position record produced by
tools/extract_quiet.py, the 22 extra features consumed by
tools/train_v3.py and mirrored exactly by src/eval.rs:

  idx  feature (all are White-perspective: white value - black value)
  ---  ---------------------------------------------------------------
  0-5  passed pawns by relative rank (pawn on its 2nd..7th rank)
  6    isolated pawns (count)
  7    doubled pawns (extra pawns beyond the first per file)
  8    backward pawns (count; definition below)
  9    king pawn shield (own pawns on the 6 squares in front of the
       king: the 3 directly in front + the 3 two ranks in front)
  10-13 king-zone attacks by piece type N, B, R, Q: number of squares
       in the enemy king's zone (king square + 8 neighbours) attacked
       by pieces of that type (union of their attack sets)
  14-17 mobility by piece type N, B, R, Q: number of squares in the
       union of that type's pseudo-legal attack sets, excluding
       squares occupied by the moving side itself
  18   rooks on open files (no pawns of either colour on the file)
  19   rooks on semi-open files (no own pawns, at least one enemy)
  20   rooks on the 7th relative rank
  21   knight outposts: knight on relative ranks 4-6, defended by a
       friendly pawn, and no enemy pawn on an adjacent file positioned
       to chase it (enemy pawn strictly ahead of the knight's rank can
       still advance to attack its square)

Backward pawn definition (per colour): a pawn whose stop square (the
square directly in front) is attacked by an enemy pawn, and which has
no friendly pawn on an adjacent file either on its own rank (such a
pawn defends the stop square) or one rank behind it (such a pawn
defends the pawn itself).

This module is the reference implementation: tools/train_v3.py's
feature matrices come from these numbers, and tests/uci_smoke.py
recomputes them on demand for the Rust/Python parity check.

Usage:
  python3 tools/quiet_features.py            # all qshard bins -> .npy
  python3 tools/quiet_features.py --check    # symmetry sanity checks
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import chess

ROOT = Path(__file__).resolve().parent.parent
N_EXTRA = 22
CODES = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
         chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def board_from_record(rec):
    board = chess.Board(fen=None)
    board.clear()
    for sq in range(64):
        code = int(rec[sq])
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            board.set_piece_at(sq, chess.Piece((code - 1) % 6 + 1, color))
    board.turn = chess.WHITE if rec[64] == 0 else chess.BLACK
    return board


def _pawns(board, color):
    return list(board.pieces(chess.PAWN, color))


def _passed(board, color):
    """Passed-pawn counts by bonus index (relative rank 2..7 -> 0..5)."""
    out = [0] * 6
    enemy = board.pieces(chess.PAWN, not color)
    enemy_fr = [(chess.square_file(s), chess.square_rank(s)) for s in enemy]
    for sq in _pawns(board, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if color == chess.WHITE:
            ahead = any(abs(ef - f) <= 1 and er > r for ef, er in enemy_fr)
            rel = r  # rank number (1-based from White side) = r+1
            idx = rel - 1
        else:
            ahead = any(abs(ef - f) <= 1 and er < r for ef, er in enemy_fr)
            idx = 6 - r
        if not ahead and 0 <= idx <= 5:
            out[idx] += 1
    return out


def _isolated(board, color):
    files = {chess.square_file(s) for s in _pawns(board, color)}
    n = 0
    for sq in _pawns(board, color):
        f = chess.square_file(sq)
        if (f - 1) not in files and (f + 1) not in files:
            n += 1
    return n


def _doubled(board, color):
    counts = {}
    for sq in _pawns(board, color):
        f = chess.square_file(sq)
        counts[f] = counts.get(f, 0) + 1
    return sum(c - 1 for c in counts.values())


def _backward(board, color):
    own = {(chess.square_file(s), chess.square_rank(s))
           for s in _pawns(board, color)}
    enemy = {(chess.square_file(s), chess.square_rank(s))
             for s in board.pieces(chess.PAWN, not color)}
    n = 0
    for sq in _pawns(board, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if color == chess.WHITE:
            stop_attacked = (f - 1, r + 2) in enemy or (f + 1, r + 2) in enemy
            support = ((f - 1, r) in own or (f + 1, r) in own
                       or (f - 1, r - 1) in own or (f + 1, r - 1) in own)
        else:
            stop_attacked = (f - 1, r - 2) in enemy or (f + 1, r - 2) in enemy
            support = ((f - 1, r) in own or (f + 1, r) in own
                       or (f - 1, r + 1) in own or (f + 1, r + 1) in own)
        if stop_attacked and not support:
            n += 1
    return n


def _shield(board, color):
    ksq = board.king(color)
    if ksq is None:
        return 0
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    step = 1 if color == chess.WHITE else -1
    own = set(_pawns(board, color))
    n = 0
    for d in (step, 2 * step):
        for df in (-1, 0, 1):
            f, r = kf + df, kr + d
            if 0 <= f <= 7 and 0 <= r <= 7:
                if chess.square(f, r) in own:
                    n += 1
    return n


def _zone_attacks(board, color):
    """Per-type (N,B,R,Q) count of squares in `color`'s king zone
    attacked by the ENEMY of `color`."""
    ksq = board.king(color)
    if ksq is None:
        return [0, 0, 0, 0]
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    zone = set()
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                zone.add(chess.square(f, r))
    out = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        attacked = set()
        for sq in board.pieces(pt, not color):
            attacked |= set(board.attacks(sq))
        out.append(len(zone & attacked))
    return out


def _mobility(board, color):
    own_occ = set(chess.SquareSet(board.occupied_co[color]))
    out = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        attacked = set()
        for sq in board.pieces(pt, color):
            attacked |= set(board.attacks(sq))
        out.append(len(attacked - own_occ))
    return out


def _rook_files(board, color):
    pawn_files_any = {chess.square_file(s)
                      for s in board.pieces(chess.PAWN, chess.WHITE)}
    pawn_files_any |= {chess.square_file(s)
                       for s in board.pieces(chess.PAWN, chess.BLACK)}
    own_files = {chess.square_file(s) for s in _pawns(board, color)}
    open_n = semi_n = 0
    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        if f not in pawn_files_any:
            open_n += 1
        elif f not in own_files:
            semi_n += 1
    return open_n, semi_n


def _rook_seventh(board, color):
    want = 6 if color == chess.WHITE else 1
    return sum(1 for sq in board.pieces(chess.ROOK, color)
               if chess.square_rank(sq) == want)


def _outposts(board, color):
    # A square is pawn-defended if a friendly pawn attacks it.
    defended = set()
    for sq in _pawns(board, color):
        defended |= set(board.attacks(sq))
    enemy_fr = [(chess.square_file(s), chess.square_rank(s))
                for s in board.pieces(chess.PAWN, not color)]
    n = 0
    for sq in board.pieces(chess.KNIGHT, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if color == chess.WHITE:
            if r not in (3, 4, 5):
                continue
            chase = any(abs(ef - f) == 1 and er > r for ef, er in enemy_fr)
        else:
            if r not in (2, 3, 4):
                continue
            chase = any(abs(ef - f) == 1 and er < r for ef, er in enemy_fr)
        if sq in defended and not chase:
            n += 1
    return n


def compute_features(board):
    """The 22 extra features of a position, White perspective."""
    f = np.zeros(N_EXTRA, dtype=np.float64)
    pw, pb = _passed(board, chess.WHITE), _passed(board, chess.BLACK)
    for i in range(6):
        f[i] = pw[i] - pb[i]
    f[6] = _isolated(board, chess.WHITE) - _isolated(board, chess.BLACK)
    f[7] = _doubled(board, chess.WHITE) - _doubled(board, chess.BLACK)
    f[8] = _backward(board, chess.WHITE) - _backward(board, chess.BLACK)
    f[9] = _shield(board, chess.WHITE) - _shield(board, chess.BLACK)
    # Zone attacks ON each king by the enemy; feature = pressure White
    # exerts on Black's king minus pressure Black exerts on White's.
    on_black = _zone_attacks(board, chess.BLACK)
    on_white = _zone_attacks(board, chess.WHITE)
    for i in range(4):
        f[10 + i] = on_black[i] - on_white[i]
    mw, mb = _mobility(board, chess.WHITE), _mobility(board, chess.BLACK)
    for i in range(4):
        f[14 + i] = mw[i] - mb[i]
    ow, sw = _rook_files(board, chess.WHITE)
    ob, sb = _rook_files(board, chess.BLACK)
    f[18] = ow - ob
    f[19] = sw - sb
    f[20] = _rook_seventh(board, chess.WHITE) - _rook_seventh(board, chess.BLACK)
    f[21] = _outposts(board, chess.WHITE) - _outposts(board, chess.BLACK)
    return f


def _work(records):
    return np.stack([compute_features(board_from_record(r)) for r in records])


def process_file(bin_path, workers):
    raw = np.fromfile(bin_path, dtype=np.uint8)
    assert raw.size % 66 == 0, bin_path
    recs = raw.reshape(-1, 66)
    chunks = np.array_split(recs, max(1, len(recs) // 20_000))
    with Pool(workers) as pool:
        parts = pool.map(_work, [c for c in chunks if len(c)])
    ftr = np.concatenate(parts).astype(np.float32)
    out = bin_path.replace(".bin", "-ftr.npy")
    np.save(out, ftr)
    print(f"{bin_path}: {len(recs)} records -> {out} {ftr.shape}",
          flush=True)
    return out


def check():
    start = chess.Board()
    f = compute_features(start)
    print("startpos features:", f.astype(int))
    assert np.all(f == 0), "startpos must be perfectly symmetric"
    # A lone passed pawn: White pawn e5, kings far away.
    b = chess.Board("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
    f = compute_features(b)
    print("e5-pawn features:  ", f.astype(int))
    assert f[3] == 1 and f.sum() >= 1  # rank-5 pawn -> idx 3
    print("checks OK")


def main():
    if "--check" in sys.argv:
        check()
        return
    workers = int(os.environ.get("FTR_WORKERS", "3"))
    pattern = str(ROOT / "data" / "positions_quiet" / "qshard*.bin")
    for bin_path in sorted(glob.glob(pattern)):
        process_file(bin_path, workers)


if __name__ == "__main__":
    main()
