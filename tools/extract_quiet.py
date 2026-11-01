#!/usr/bin/env python3
"""Proty V2 corpus extractor — QUIET-filter variant for the 2.1.0 retune.

Recovers the elite corpus downloaded for Proty V1 (TWIC + PGN Mentor,
all batches) and extracts Texel-training positions for the Anciency-
style tuner. Labels are GAME RESULTS ONLY — no engine labels anywhere.

Filter (widened roster, user decision 2026-10-31):
  A player passes if (a) their normalised name is on the FIDE title
  list with a title in {GM, WGM, IM, WIM, FM, WFM, CM, WCM}, or
  (b) their Elo header in the game is >= 2000.
  A game qualifies iff BOTH players pass AND, when both Elo headers
  are present, min(WhiteElo, BlackElo) >= 2000 (the WCM floor). When
  an Elo header is missing, that player must pass via (a). When in
  doubt the game is excluded. All results kept.

Extraction (Anciency's rules, tools/train.py of that project):
  mainline positions from ply 6, side to move not in check, plain
  material balance within 1000 cp (100/320/330/500/1000), games of at
  least 20 plies, at most 16 positions sampled per game (even stride).
  Records: 66 bytes — 64 square bytes (python-chess square order,
  a1=0; 0 empty, 1..6 white P..K, 7..12 black P..K), turn byte
  (0 white, 1 black), result byte from the SIDE-TO-MOVE point of view
  (2 = stm won, 1 = draw, 0 = stm lost).
  Split by game-hash bucket (1 of 20 -> validation), as V1 did.
  Dedupe on (surnames, result, movetext), as V1 did.

Also accumulates the opening book exactly like Anciency's
tools/make_book.py (first 20 plies, result-weighted 3/2/1).

Modes:
  (default)     sample positions until --target total positions.
  --census      header-only pass over the shard's archives: counts
                seen / filter-passing games without parsing movetext.
"""
import argparse, hashlib, io, json, os, re, sqlite3, sys, time, zipfile
from collections import defaultdict

import chess
import chess.pgn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TITLES = {"GM", "WGM", "IM", "WIM", "FM", "WFM", "CM", "WCM"}
TITLE_RANK = {"CM": 1, "WCM": 1, "FM": 2, "WFM": 2, "IM": 3, "WIM": 3,
              "GM": 4, "WGM": 4}
ELO_PASS = 2000
ELO_FLOOR = 2000
SKIP_PLIES = 6
MATERIAL_CAP = 1000
PER_GAME_CAP = 16
MIN_PLIES = 20
BOOK_PLIES = 20
MAT = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 1000, chess.KING: 0}
GAME_BATCHES = ["batch-0001-twic-0920-1169", "batch-0002-twic-1170-1419",
                "batch-0003-twic-1420-1668", "batch-0004-pgnmentor-a-m",
                "batch-0005-pgnmentor-n-z"]


def norm_name(name):
    """'Carlsen, Magnus' / 'Magnus Carlsen' -> (surname_lower, first_initial)"""
    name = (name or "").strip()
    if "," in name:
        sur, first = name.split(",", 1)
        sur = sur.strip().lower()
        first = first.strip()
    else:
        parts = name.split()
        if len(parts) < 2:
            return (name.lower(), "")
        sur = parts[-1].lower()
        first = parts[0]
    sur = re.sub(r"[^a-zà-ÿ' \-]", "", sur)
    return (sur, first[0].lower() if first else "")


def load_fide_titles(zip_path):
    titles = {}
    if not zip_path or not os.path.exists(zip_path):
        print("WARNING: FIDE list not found, title-list rule (a) disabled")
        return titles
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".txt")]
        raw = z.read(names[0]).decode("utf-8", "replace")
    n_titled = 0
    for line in raw.splitlines()[1:]:
        # Fixed-width layout verified against the Oct-2026 list: name at
        # cols 15..75, Tit at 82..87, WTit at 87..93, OTit at 93..99.
        if len(line) < 86:
            continue
        name = line[15:75].strip()
        cands = [line[82:87].strip(), line[87:93].strip(), line[93:99].strip()]
        best = max([TITLE_RANK[c] for c in cands if c in TITLES] or [0])
        if best and name:
            key = norm_name(name)
            if best > titles.get(key, 0):
                titles[key] = best
            n_titled += 1
    print(f"FIDE titled players (CM+): {n_titled} entries, "
          f"{len(titles)} unique names", flush=True)
    return titles


def list_archives():
    """All game-archive files, interleaved round-robin across batches so
    a partial run still draws from every batch."""
    per_batch = []
    for b in GAME_BATCHES:
        d = os.path.join(ROOT, "data", "raw_batches", b)
        files = []
        for fname in sorted(os.listdir(d)):
            if fname.startswith("_"):
                continue
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                files.append((b, fpath))
        per_batch.append(files)
    out = []
    i = 0
    while any(i < len(f) for f in per_batch):
        for f in per_batch:
            if i < len(f):
                out.append(f[i])
        i += 1
    return out


def elo_of(headers, key):
    v = headers.get(key, "")
    return int(v) if v and v.isdigit() else None


def qualifies(headers, titles):
    result = headers.get("Result", "*")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return None
    wn, bn = headers.get("White", "?"), headers.get("Black", "?")
    we, be = elo_of(headers, "WhiteElo"), elo_of(headers, "BlackElo")

    def passed(name, elo):
        if norm_name(name) in titles:
            return True
        return elo is not None and elo >= ELO_PASS

    if not (passed(wn, we) and passed(bn, be)):
        return None
    if we is not None and be is not None and min(we, be) < ELO_FLOOR:
        return None
    return wn, bn, we, be, result


def encode_record(board, result_white):
    squares = bytearray(66)
    for sq, piece in board.piece_map().items():
        code = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        squares[sq] = code
    squares[64] = 0 if board.turn == chess.WHITE else 1
    stm = result_white if board.turn == chess.WHITE else 1.0 - result_white
    squares[65] = int(round(stm * 2))
    return bytes(squares)


HDR_RE = {k: re.compile(r'\[' + k + r'\s+"([^"]*)"\]') for k in
          ("White", "Black", "WhiteElo", "BlackElo", "Result")}


def census_archive(fpath, titles, stats):
    def feed(text):
        for chunk in re.split(r"\n\[Event ", text):
            stats["games_seen"] += 1
            hdrs = {}
            for k, rx in HDR_RE.items():
                m = rx.search(chunk)
                if m:
                    hdrs[k] = m.group(1)
            if qualifies(hdrs, titles):
                stats["qualified_headers"] += 1

    if fpath.endswith(".zip"):
        try:
            with zipfile.ZipFile(fpath) as z:
                for inner in z.namelist():
                    if inner.lower().endswith(".pgn"):
                        feed(z.read(inner).decode("utf-8", "replace"))
        except zipfile.BadZipFile:
            stats["bad_zip"] += 1
    elif fpath.lower().endswith(".pgn"):
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            feed(fh.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--target", type=int, default=1_600_000,
                    help="total train+val positions for this shard")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="output file tag (default qshard<N>)")
    args = ap.parse_args()

    fide = os.path.join(ROOT, "data", "raw_batches", "batch-0006-fide-titles",
                        "standard_rating_list.zip")
    titles = load_fide_titles(fide)
    archives = [a for i, a in enumerate(list_archives())
                if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(archives)} archives",
          flush=True)

    outdir = os.path.join(ROOT, "data", "positions_quiet")
    os.makedirs(outdir, exist_ok=True)
    tag = args.tag or f"qshard{args.shard}"

    if args.census:
        stats = {"games_seen": 0, "qualified_headers": 0, "bad_zip": 0,
                 "per_batch": {}}
        t0 = time.time()
        for batch, fpath in archives:
            before = stats["qualified_headers"]
            census_archive(fpath, titles, stats)
            b = stats["per_batch"].setdefault(batch, {"qualified": 0})
            b["qualified"] += stats["qualified_headers"] - before
            print(f"  census {os.path.basename(fpath)}: seen "
                  f"{stats['games_seen']}, qualified "
                  f"{stats['qualified_headers']}, {time.time()-t0:.0f}s",
                  flush=True)
        stats["shard"] = args.shard
        stats["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(outdir, f"census-{tag}.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print(json.dumps(stats, indent=1))
        return

    db = sqlite3.connect(os.path.join(outdir, f"dedup-{tag}.sqlite"))
    db.execute("CREATE TABLE IF NOT EXISTS games (h TEXT PRIMARY KEY)")
    train_f = open(os.path.join(outdir, f"{tag}-train.bin"), "ab")
    val_f = open(os.path.join(outdir, f"{tag}-val.bin"), "ab")
    book = defaultdict(lambda: defaultdict(int))
    stats = {"games_seen": 0, "qualified": 0, "dup": 0, "reject_filter": 0,
             "reject_short": 0, "reject_unfinished": 0, "positions": 0,
             "positions_val": 0, "parse_errors": 0, "reject_nonquiet": 0,
             "tier_min_rank": {}, "elo_only_games": 0, "book_games": 0}
    t0 = time.time()
    done = False

    def process_stream(stream):
        nonlocal done
        while not done:
            try:
                game = chess.pgn.read_game(stream)
            except Exception:
                stats["parse_errors"] += 1
                break
            if game is None:
                break
            stats["games_seen"] += 1
            q = qualifies(game.headers, titles)
            if q is None:
                # distinguish unfinished-result rejects for the report
                if game.headers.get("Result", "*") not in (
                        "1-0", "0-1", "1/2-1/2"):
                    stats["reject_unfinished"] += 1
                else:
                    stats["reject_filter"] += 1
                continue
            wn, bn, we, be, result = q
            result_white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
            board = game.board()
            moves_san = []
            candidates = []
            plies = 0
            # Collect the mainline first: the quiet filter needs the
            # move played FROM each candidate position.
            mainline = []
            node = game
            while node.variations:
                node = node.variations[0]
                mainline.append(node.move)
            for i, mv in enumerate(mainline):
                # book: record the position BEFORE the move, first 20 plies
                if plies < BOOK_PLIES:
                    mover_white = board.turn == chess.WHITE
                    if result == "1/2-1/2":
                        add = 2
                    elif (result == "1-0") == mover_white:
                        add = 3
                    else:
                        add = 1
                    fen_parts = board.fen().split()
                    book[" ".join(fen_parts[:3])][mv.uci()] += add
                moves_san.append(board.san(mv))
                board.push(mv)
                plies += 1
                if plies < SKIP_PLIES:
                    continue
                if board.is_check():
                    continue
                bal = 0
                for pc in board.piece_map().values():
                    bal += MAT[pc.piece_type] * (
                        1 if pc.color == chess.WHITE else -1)
                if abs(bal) > MATERIAL_CAP:
                    continue
                # QUIET FILTER: the move actually played from this
                # position must exist and be quiet — no capture, no
                # promotion, no check given.
                if i + 1 >= len(mainline):
                    stats["reject_nonquiet"] += 1
                    continue
                nxt = mainline[i + 1]
                if (board.is_capture(nxt) or nxt.promotion is not None
                        or board.gives_check(nxt)):
                    stats["reject_nonquiet"] += 1
                    continue
                candidates.append(encode_record(board, result_white))
            if plies < MIN_PLIES:
                stats["reject_short"] += 1
                continue
            key = hashlib.sha1(
                (norm_name(wn)[0] + "|" + norm_name(bn)[0] + "|" + result +
                 "|" + " ".join(moves_san)).encode("utf-8", "replace")
            ).hexdigest()
            cur = db.execute("SELECT 1 FROM games WHERE h=?", (key,))
            if cur.fetchone():
                stats["dup"] += 1
                continue
            db.execute("INSERT INTO games VALUES (?)", (key,))
            stats["qualified"] += 1
            stats["book_games"] += 1
            tw = titles.get(norm_name(wn), 0)
            tb = titles.get(norm_name(bn), 0)
            tier = str(min(tw, tb))
            stats["tier_min_rank"][tier] = \
                stats["tier_min_rank"].get(tier, 0) + 1
            if tw == 0 or tb == 0:
                stats["elo_only_games"] += 1
            if len(candidates) > PER_GAME_CAP:
                idx = [round(i * (len(candidates) - 1) / (PER_GAME_CAP - 1))
                       for i in range(PER_GAME_CAP)]
                candidates = [candidates[j] for j in idx]
            bucket = int(key[:8], 16) % 20
            out = val_f if bucket == 0 else train_f
            for r in candidates:
                out.write(r)
            stats["positions"] += len(candidates)
            if bucket == 0:
                stats["positions_val"] += len(candidates)
            if stats["qualified"] % 2000 == 0:
                db.commit()
                train_f.flush(); val_f.flush()
                print(f"  [{tag}] {stats['games_seen']} seen, "
                      f"{stats['qualified']} qualified, "
                      f"{stats['positions']} pos, {time.time()-t0:.0f}s",
                      flush=True)
            total = stats["positions"]
            if total >= args.target:
                done = True

    for batch, fpath in archives:
        if done:
            break
        fname = os.path.basename(fpath)
        if fname.endswith(".zip"):
            try:
                with zipfile.ZipFile(fpath) as z:
                    for inner in z.namelist():
                        if done:
                            break
                        if inner.lower().endswith(".pgn"):
                            with z.open(inner) as raw:
                                text = io.TextIOWrapper(
                                    raw, encoding="utf-8", errors="replace")
                                process_stream(text)
            except zipfile.BadZipFile:
                print(f"  BAD ZIP skipped: {fname}")
        elif fname.lower().endswith(".pgn"):
            with open(fpath, encoding="utf-8", errors="replace") as text:
                process_stream(text)
        print(f"  [{tag}] finished archive {fname} "
              f"({stats['positions']} positions)", flush=True)

    train_f.close(); val_f.close(); db.commit(); db.close()
    with open(os.path.join(outdir, f"book-{tag}.json"), "w") as f:
        json.dump({k: dict(v) for k, v in book.items()}, f)
    stats["shard"] = args.shard
    stats["seconds"] = round(time.time() - t0, 1)
    stats["filter_rule"] = (
        "both players: FIDE title in {GM,WGM,IM,WIM,FM,WFM,CM,WCM} by "
        "name OR Elo>=2000; if both Elo headers present, min Elo >= 2000; "
        "QUIET: stm not in check and the move played from the position "
        "is not a capture, promotion or check")
    with open(os.path.join(outdir, f"extract-{tag}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
