# Proty V2 — Changelog

0.1.0–0.5.0 shipped 2026-10-31; 0.6.0, 0.7.0 and 0.7.5 shipped 2026-11-01. Newest first. Every change after
0.1.0 was kept only if the full regression suite stayed green **and**
it passed a match gate against the previous version (fixed-length
55% gates up to 0.4.0; SPRT from 0.5.0).

## [0.7.5] — 2026-11-01 — Warm-start retune on all-position data

Commissioned after the Tjj7zv8v post-mortem (Proty misevaluated
sharp positions it had never trained on; the 0.3.0 tune saw quiet
positions only). No code changes: search, feature set, book and
Syzygy support are identical to 0.7.0. Only the 826 evaluation
parameters moved.

### Changed
- Parameters re-polished by a warm-start Gauss-Newton fit
  (tools/train_v5.py): starts AT the shipped 0.3.0–0.7.0
  parameters and optimises a proximal objective — train loss
  plus (mu/2)·mean squared drift, mu = 1e-5 — on the v4
  all-position dataset (tools/extract_v4.py: same elite game
  filter and sampling as the original extractor, NO quiet
  filter; 9,600,020 positions extracted, 9,117,340 train /
  482,680 val after the split). Final: train 0.153634,
  val 0.153599 (shipped params: 0.154778 / 0.154646),
  RMS drift from the shipped parameters 3.90 cp.

### Rejected on the way
- The from-scratch all-position refit (tools/train_v4.py) won
  the loss comparison (val 0.150933) and lost the gate badly:
  owner-stopped at 70 games, 30.7%, LLR −1.716. Third instance
  of the project pattern (after 0.1.0's tune and E4): a better
  result-prediction loss does not imply better move ranking.
  A v3/v4 blend sweep confirmed the val loss is monotonic in
  the blend factor — it cannot rank these candidates at all.

### Gate (vs 0.7.0, SPRT harness, 200-game cap set by the owner)
- 200 games, W96 D42 L62 = 58.5%, LLR +2.127, verdict: cap
  (the +2.944 certification bound was not crossed).
- Kept on the standing as an UNCERTIFIED keep — the D1
  precedent (kept at 55.8%, LLR +1.154). Regression green:
  cargo tests 14/14, UCI smoke.
- Sanity record: the candidate still plays Bb6 in the
  move-10 autopsy position (score +53). The retune improved
  general judgment; it did not cure the king-pressure blind
  spot. That is 0.8.0's commissioned brief (nonlinear
  king-attack table features).

## [0.7.0] — 2026-11-01 — Phase F: endgame (Syzygy tablebases + scale factors)

### Added
- Syzygy tablebase support via shakmaty-syzygy 0.25.3 (the
  release built against this engine's shakmaty 0.27; the crate's
  0.27.0 release targets shakmaty 0.29 and cannot link here):
  - `SyzygyPath` UCI string option (`;`-separated, plus
    `:`-separated on non-Windows) loading .rtbw/.rtbz
    directories. No tables loaded = behaviour identical to
    0.6.0 (verified: startpos depth 11 = 1,420,373 nodes).
  - WDL probes in the main search: an exact verdict replaces the
    heuristic subtree at non-root nodes in table range. The
    probe accounts for the halfmove clock (cursed wins /
    blessed losses score correctly under the fifty-move rule).
    Tablebase wins score ±90,000, ply-adjusted, below the mate
    band; the probe runs after terminal detection so mates keep
    their mate scores.
  - DTZ-optimal root play: root positions in table range are
    answered directly with the tablebase's recommended move
    (depth-0 info line with the exact verdict), which converts
    won endings inside the fifty-move budget.
  - tests/tablebase_smoke.py: 8 checks against a real 3-4-5-man
    set (option listed, WDL signs, draws, in-search probe,
    KQvK conversion to mate in 9 plies). All pass.
- Endgame scale factors in the evaluation (hand-set, narrow
  material patterns only):
  - Wrong-coloured bishop + a single rook pawn vs a bare king:
    score 0 when the defending king blockades the pawn's file,
    halved otherwise.
  - Opposite-coloured bishops with no other pieces: score
    halved.

### Fixed during development (never shipped)
- The crate's best_move reports the successor position's DTZ
  from the opponent's perspective; the first build read the
  sign at face value and scored KQ-vs-K as a loss. Caught by
  the tablebase smoke suite and negated at the loader boundary
  before release.

### Notes
- This phase was commissioned as four named additions and is
  verified by correctness tests, not a match gate: tablebase
  positions essentially never arise from the gate opening
  suites. The scale factors are unit-tested on constructed
  positions only — recorded plainly, unlike the gated
  Phase A–E changes.
- Tables are not in the engine zip: the 3-4-5-man set (290
  files, 939 MiB, every file size-verified against the
  Lichess mirror) ships as a separate archive,
  syzygy-5man.zip. The same loader probes 6/7-man sets
  unchanged if pointed at them.
- Parameters and opening book unchanged from 0.6.0.

## [0.6.0] — 2026-11-01 — Phase E: search experiments, one keeper

### Added
- SEE pruning in the main search (E1): at depth <= 2, not in check,
  with a non-mate window and at least one move already searched,
  moves whose static exchange evaluation falls below -100 (depth 1)
  or -250 (depth 2) are skipped outright — captures and quiets
  alike. Gate vs 0.5.0 (SPRT, 200 ms/move, paired openings):
  accept H1 at 100 games — 64.0% (W45 D38 L17), LLR +3.070 — the
  first fully SPRT-certified gate in the project's history.
  Startpos depth-11 tree: 2,094,501 -> 1,420,373 nodes.

### Reverted (gated, measured, not shipped)
- E2 singular extensions (TT-move verification search at half
  depth, extend +1 when all alternatives fail low by 2xdepth):
  SPRT accept H0 at 100 games vs the E1 build — 36.0% (W18 D37
  L45), LLR -3.075. Reverted; tree verified bit-identical to E1
  (startpos depth 11 = 1,420,373 nodes).
- E3 countermove heuristic, solo (64x64 table keyed by the
  previous move, stored on quiet beta cutoffs, ordered just below
  the second killer): 44.8%, LLR -1.182 at the owner-set 126-game
  stop vs the E1 build; a restart-killed first run agreed (42.5%
  at 60 games). Reverted; tree verified bit-identical to E1.
- E4 2200+ subset retune (all 826 parameters retrained on
  2,900,017 quiet positions from games where both players were
  rated >= 2200): fit its own validation split better (0.14227 vs
  0.14590 for the 0.3.0 parameters) but lost the fixed-depth-6
  gate 19.0-21.0 (47.5%) against the 55% bar. Not shipped; the
  0.3.0 parameter set is unchanged.

### Notes
- Gates this round ran with an owner-set stop at 125 games
  (watcher fired at 126); E1 and E2 reached certified SPRT
  verdicts inside it.
- E4's extraction proved the shipped book is a corpus-floor
  artifact: re-merging the original 2000-floor fragments
  reproduces data/book.json's 408,841 positions exactly, while
  the 2200+ fragments would merge to a different 355,534-position
  book. Book unchanged in 0.6.0.
- Rejected E2/E3/E4 binaries are preserved under gate-opponents/
  as gate records.

## [0.5.0] — 2026-10-31 — Phase D: speed, SPRT gates

### Added
- SPRT match harness (`tests/match_sprt.py`): paired openings with
  colours swapped, pentanomial log-likelihood ratio, elo0 = 0,
  elo1 = 15, alpha = beta = 0.05, 500-game cap. This replaces the
  fixed-length 55% gates as the standing gate protocol.
- SEE capture ordering in the main search (D2): captures with
  SEE >= 0 are ordered by exchange value above the killers;
  captures with SEE < 0 are demoted below the killers.

### Changed
- Evaluation rewritten to be allocation-free (D1, bit-exact):
  per-type attack unions are computed once and shared between the
  king-zone and mobility features (previously twice), pawn
  structure features are served from a validated per-thread cache,
  the game phase is folded into the PST pass, and the Zobrist key
  is threaded through the search instead of recomputed. Bench node
  counts are identical to 0.4.0 (8,202,559 at depth 11); tuned
  bench speed on the build VM rose ~485k -> ~800k nps (+66%).
  Parity verified against the preserved reference implementation
  on >10,000 positions under both parameter sets.
- MVV-LVA capture scoring removed (superseded by SEE ordering).
  D2's fixed-depth tree is 13.6% smaller (7,087,774 nodes).

### Feats
- Gate D1 vs 0.4.0 (SPRT, 200 ms/move): stopped at 200 games at
  the owner's instruction — 55.8%, LLR +1.154. The certification
  bound (+2.944) was not crossed; D1 was kept on its positive
  standing by owner decision.
- Gate D2 vs the D1 build (SPRT, 200 ms/move): stopped at 200
  games at the owner's instruction — 58.5% (W 83, D 69, L 48),
  LLR +2.771, one pair short of the bound; kept on its standing.
  A first D2 run was killed at game 24 by a build-VM restart and
  rerun cleanly in full; the partial run counts for nothing.
- Regression for both batches: cargo tests 10/10, UCI smoke
  (incl. perft and Python-eval agreement), stalemate traps 6/6,
  time-abort suite, SMP smokes 6/6.

## [0.4.0] — 2026-10-31 — Phase C: Lazy SMP multithreading

### Added
- Lazy SMP parallel search behind the UCI `Threads` option
  (1–16, default 1). Helper threads search the same root position
  with diversified (rotated) root move lists; the main thread alone
  reports info lines and picks the move.
- Shared lockless transposition table: Stockfish-style `key ^ data`
  atomic slot pairs, the whole entry (score, depth, flag, best move)
  packed into a single 64-bit word.
- Per-thread killer and history tables.
- `Threads=1` reproduces 0.3.0 behaviour exactly.

### Feats
- Sanity match, Threads=1 vs 0.3.0: 12–8 (60%).
- Gate, Threads=2 vs 0.3.0: **27–13 (67.5%)**, 40 games at 200 ms —
  kept. (A first gate attempt was discarded as invalid — the harness
  stopped at 20 games — and rerun cleanly in full.)
- On the owner's laptop at Threads=8, the startpos depth-12
  main-thread tree halved: 3,316,888 → 1,668,296 nodes.
- Regression: 8/8 unit tests (including new shared-TT pack/probe
  tests), perft exact, stalemate traps 6/6, clock suite, 6/6 SMP
  smokes.
- Known caveat: on the CPU-throttled build VM, Threads=2 wall-clock
  speed-up measures flat (0.91–1.00×) despite ~30% fewer main-thread
  nodes; the scaling shows on real hardware (above).

## [0.3.0] — 2026-10-31 — Phase B: structural eval + quiet retune

### Added
- 22 structural evaluation features (44 new tapered parameters,
  782 → 826 total): passed pawns by relative rank, isolated,
  doubled and backward pawns, king pawn shield, king-zone attacks
  received per attacker type (N/B/R/Q), mobility per piece type,
  rooks on open and semi-open files, rooks on the seventh rank,
  knight outposts.
- Full retune on 2,900,016 quiet positions. Quiet-validation loss
  0.147242, against 0.15533 for the handcrafted values and 0.15092
  for the 0.1.0 tuned set.

### Feats
- Gate vs 0.2.0: fixed depth **28–12 (70%)**, timed confirmation
  **11.5–8.5 (57.5%)** — kept.
- Overall vs 0.1.0, timed: **34.5–5.5 (86.25%)**.
- First tuned parameter set to beat its own handcrafted baseline
  head-to-head: 14.5–5.5 (72.5%).
- The richer eval costs node rate — clean bench NPS fell to ~400k —
  but the search trees shrank so much that time-to-depth improved
  roughly 8×.

## [0.2.0] — 2026-10-31 — Phase A: search upgrades

### Added
- Batch A2: null-move pruning, late-move reductions, futility
  pruning.
- Batch A3: aspiration windows, check extensions, quiescence
  SEE/delta pruning.

### Reverted
- Batch A1 (SEE capture ordering + countermove): 53.75% combined,
  under the 55% gate — reverted in full. The batch mixed two changes,
  so which half failed was never isolated.

### Feats
- A2 gate vs 0.1.0, timed: **27–13 (67.5%)** — kept.
- A3 gate vs the A2 build, timed: **31–9 (77.5%)** — kept.
- Clean bench NPS 1,070,844.

## [0.1.0] — 2026-10-31 — Initial release

### Added
- Handcrafted tapered evaluation — material, piece-square tables,
  bishop pair; 782 parameters — Texel-tuned on game results only.
  No NNUE, no Stockfish labels, no self-play, by permanent design.
- Tuning corpus: 4,152,424 elite games filtered to 3,047,160
  qualified games (FIDE title GM–WCM on a 23,804-name list, or
  Elo ≥ 2000), of which 200,046 games produced 3,200,008 training
  positions. Validation loss: 0.16188 handcrafted → 0.15279 tuned.
- Opening book: 408,841 positions from the same corpus, moves
  weighted by frequency × result.
- Negamax alpha-beta search with quiescence, transposition table,
  MVV-LVA + killer + history ordering, and hardened clock handling
  (completed-iteration rule, hard node failsafe, per-node stop
  polling — the Proty V1 time-abort lessons, applied from birth).
- UCI options: `Baseline=true` selects the untuned handcrafted
  parameter set; the binary defaults to the tuned set.
- Clean bench NPS ~1.31M.

### Feats
- Beat the original Anciency **13.5–6.5** at fixed depth 6.
- Beat capped Stockfish 19 (depth 5 + 0.1 s/move) **5.5–4.5,
  undefeated**, in a 10-game exhibition with the book on
  (1 win, 9 draws; held all 5 games as Black).

### Known issue, stated plainly
- The 0.1.0 tuned set *lost* to its own untuned baseline 23–37 over
  60 games despite the better tuning loss. It shipped anyway as the
  default (the baseline remains available via `Baseline=true`), and
  fixing this became a 0.3.0 milestone — see above.
