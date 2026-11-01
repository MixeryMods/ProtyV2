#!/usr/bin/env python3
"""UCI smoke test for the native Anciency binary.

Checks: uci/uciok, isready/readyok, perft from startpos (depth 3, 4) and
from the Kiwipete middlegame (depth 3), a legal bestmove at depth 10 and
with movetime, and that the engine's `eval` command matches an
independent Python re-implementation of the trained parameters.
"""
import re
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "proty-v2"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def run_session(commands, timeout=180):
    proc = subprocess.run(
        [str(BIN)], input="\n".join(commands) + "\n",
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout.splitlines()


def parse_trained_params():
    text = (ROOT / "src" / "trained_params.inc").read_text()
    if text.strip() == "BASELINE":
        text = (ROOT / "src" / "tuned.rs").read_text().split("pub const BASELINE")[1]
    def arr(name):
        m = re.search(name + r": \[([^\]]*)\]", text)
        return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    def pst(name):
        m = re.search(name + r": \[(.*?)\n    \],", text, re.S)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        nums = [int(x) for x in re.findall(r"-?\d+", body)]
        assert len(nums) == 384
        return nums
    params = {
        "value_mg": arr("value_mg"), "value_eg": arr("value_eg"),
        "pst_mg": pst("pst_mg"), "pst_eg": pst("pst_eg"),
        "bp_mg": int(re.search(r"bishop_pair_mg: (-?\d+)", text).group(1)),
        "bp_eg": int(re.search(r"bishop_pair_eg: (-?\d+)", text).group(1)),
    }
    # Phase B structural-feature parameters (2.1.0+). Parsed
    # generically; missing fields (pre-2.1.0 params) default to zero.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qf", ROOT / "tools" / "quiet_features.py")
    qf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qf)
    params["_qf"] = qf
    for name in ("passed_mg", "passed_eg", "king_attack_mg",
                 "king_attack_eg", "mobility_mg", "mobility_eg"):
        m = re.search(name + r": \[([^\]]*)\]", text)
        params[name] = ([int(x) for x in m.group(1).replace("\n", " ")
                         .split(",") if x.strip()] if m else [])
    for name in ("isolated_mg", "isolated_eg", "doubled_mg", "doubled_eg",
                 "backward_mg", "backward_eg", "shield_mg", "shield_eg",
                 "rook_open_mg", "rook_open_eg", "rook_semi_mg",
                 "rook_semi_eg", "rook_seventh_mg", "rook_seventh_eg",
                 "outpost_mg", "outpost_eg"):
        m = re.search(name + r": (-?\d+)", text)
        params[name] = int(m.group(1)) if m else 0
    return params


def python_eval(board: chess.Board, p) -> int:
    mg = eg = 0
    phase = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}
    pw = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    for sq, piece in board.piece_map().items():
        ri = piece.piece_type - 1
        sign = 1 if piece.color == chess.WHITE else -1
        idx = (sq ^ 56) if piece.color == chess.WHITE else sq
        mg += sign * (p["value_mg"][ri] + p["pst_mg"][ri * 64 + idx])
        eg += sign * (p["value_eg"][ri] + p["pst_eg"][ri * 64 + idx])
        phase += pw.get(piece.piece_type, 0)
        if piece.piece_type == chess.BISHOP:
            bishops[piece.color] += 1
    if bishops[chess.WHITE] >= 2:
        mg += p["bp_mg"]; eg += p["bp_eg"]
    if bishops[chess.BLACK] >= 2:
        mg -= p["bp_mg"]; eg -= p["bp_eg"]
    # Phase B structural features via the reference implementation.
    if p.get("passed_mg"):
        def pad(v, n):
            return (list(v) + [0] * n)[:n]

        def extras(suffix):
            return (pad(p[f"passed_{suffix}"], 6)
                    + [p[f"isolated_{suffix}"], p[f"doubled_{suffix}"],
                       p[f"backward_{suffix}"], p[f"shield_{suffix}"]]
                    + pad(p[f"king_attack_{suffix}"], 4)
                    + pad(p[f"mobility_{suffix}"], 4)
                    + [p[f"rook_open_{suffix}"], p[f"rook_semi_{suffix}"],
                       p[f"rook_seventh_{suffix}"], p[f"outpost_{suffix}"]])

        x = p["_qf"].compute_features(board)
        emg, eeg = extras("mg"), extras("eg")
        for i in range(22):
            mg += int(x[i]) * emg[i]
            eg += int(x[i]) * eeg[i]
    phase = min(phase, 24)
    return int((mg * phase + eg * (24 - phase)) / 24) if (mg * phase + eg * (24 - phase)) >= 0 \
        else -int(-(mg * phase + eg * (24 - phase)) / 24)


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(("PASS" if cond else "FAIL"), label, detail)
        ok = ok and cond

    lines = run_session(["uci", "isready", "quit"])
    check("uciok", "uciok" in lines)
    check("readyok", "readyok" in lines)

    lines = run_session(["perft 3", "perft 4", "quit"])
    check("perft 3 startpos", "perft 3 = 8902" in lines, str(lines))
    check("perft 4 startpos", "perft 4 = 197281" in lines)

    lines = run_session([f"position fen {KIWIPETE}", "perft 3", "quit"])
    check("perft 3 kiwipete", "perft 3 = 97862" in lines, str(lines))

    # bestmove legality at depth 10 and with movetime, book disabled for
    # the movetime case so a real search must produce the move
    lines = run_session(["position startpos", "go depth 10", "quit"], timeout=180)
    bm = [l for l in lines if l.startswith("bestmove")]
    legal = {m.uci() for m in chess.Board().legal_moves}
    check("depth10 bestmove legal", bm and bm[-1].split()[1] in legal, str(bm))

    lines = run_session(
        ["setoption name Book value false", "position fen " + KIWIPETE,
         "go movetime 500", "quit"], timeout=180)
    bm = [l for l in lines if l.startswith("bestmove")]
    board = chess.Board(KIWIPETE)
    legal = {m.uci() for m in board.legal_moves}
    check("movetime bestmove legal", bm and bm[-1].split()[1] in legal, str(bm))

    # eval consistency: Rust `eval` command vs Python re-implementation
    params = parse_trained_params()
    fens = [
        chess.STARTING_FEN,
        KIWIPETE,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    for fen in fens:
        lines = run_session([f"position fen {fen}", "eval", "quit"])
        m = [l for l in lines if l.startswith("eval ")]
        rust_val = int(m[0].split()[1]) if m else None
        py_val = python_eval(chess.Board(fen), params)
        # Rust integer division truncates toward zero, Python // floors:
        # allow the known 1cp rounding difference on odd tapered scores.
        check(f"eval match {fen[:30]}...", rust_val is not None and abs(rust_val - py_val) <= 1,
              f"rust={rust_val} python={py_val}")

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
