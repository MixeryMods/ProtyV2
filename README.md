# Proty V2 — 0.7.5

A UCI chess engine forked from **Anciency** (handcrafted evaluation,
alpha-beta search), with the evaluation **Texel-tuned on game results
only** — no neural net, no Stockfish labels, no self-play — trained on
the full modern elite corpus (GM / WGM / IM / WIM / FM / WFM / CM /
WCM, or Elo ≥ 2000). This release is a gated upgrade: every change
below was kept only if the regression suite stayed green **and** it
passed a match gate against the previous version (fixed-length
≥40-game, ≥55% gates through 0.4.0; SPRT from 0.5.0); anything that
failed was reverted.

UCI id: `Proty V2`. Just run the binary and talk UCI.

## Version mapping

| version | build |
|---|---|
| **0.1.0** | the original release, shipped at the time under the label "2.0.0" (782 tuned params, 408,841-position book) — file `proty-v2.zip` |
| **0.2.0** | Phase A: search upgrades only, 0.1.0 parameters — file `proty-v2-0.2.0.zip` |
| **0.3.0** | Phase B: Phase A search + structural eval features + quiet-position retune — file `proty-v2-0.3.0.zip` |
| **0.4.0** | Phase C: Lazy SMP multithreading on top of 0.3.0 (search and eval unchanged at Threads=1) — file `proty-v2-0.4.0.zip` |
| **0.5.0** | Phase D: bit-exact eval speed (D1) + SEE capture ordering (D2), SPRT-gated — file `proty-v2-0.5.0.zip` |
| **0.6.0** | Phase E: search experiments — only E1 (SEE pruning) ships; E2/E3/E4 gated and reverted; 0.3.0 parameters unchanged — file `proty-v2-0.6.0.zip` |
| **0.7.0** | Phase F: endgame — Syzygy tablebase probing (SyzygyPath, WDL in search, DTZ at the root) + two endgame scale factors; parameters and book unchanged — file `proty-v2-0.7.0.zip` |
| **0.7.5** | Warm-start retune: 0.7.0 code unchanged; the 826 parameters re-polished from their shipped values on 9.12M all-position samples (sharp positions included), proximal leash mu = 1e-5, RMS drift 3.90 cp; gate vs 0.7.0: 58.5% over 200 games (LLR +2.127, cap — kept uncertified) — file `proty-v2-0.7.5.zip` (this build) |

## The honest one-line summary

0.1.0's tuning improved prediction but lost at the board to its own
untuned baseline. 0.3.0 fixes both halves: a modernised search
(Phase A) and a richer evaluation retrained on quiet positions only
(Phase B). The shipped 0.3.0 beats the shipped 0.1.0 by
**34.5–5.5 (86.25%)** at 200 ms/move over 40 games (book off), and
its tuned parameters now also beat the handcrafted `Baseline=true`
set in the same binary (14.5–5.5, 72.5%). The one regression, stated
plainly: at a fixed depth of 6 the Phase A search scores only
6.0–14.0 against 0.1.0 — the pruning stack trades per-ply accuracy
for speed, which pays at any time control and costs when depth is
capped externally.

0.4.0 adds **Lazy SMP multithreading**: a `Threads` UCI option
(1–16, default 1). Helper threads run the same search on the same
position with rotated root-move orders and share one lockless
transposition table with the main thread; only the main thread
reports and picks the move, and all 0.3.0 clock behaviour is
unchanged. Threads=1 is exactly 0.3.0. The gate: Threads=2 beat
0.3.0 by **27.0–13.0 (67.5%)** at 200 ms/move over 40 games, and a
Threads=1 sanity match scored 12.0–8.0 (60%) over 20 games,
confirming the refactor did not weaken the single-thread search.
Stated plainly: this build VM is CPU-throttled, so a Threads 1 vs 2
wall-clock bench here showed no speedup (0.91–1.00×) even though the
shared table cut the main thread's node count ~30%. On the owner's
8-thread laptop, startpos depth 12 took 3,316,888 main-thread nodes
at Threads=1 vs 1,668,296 at Threads=8 — the tree halves; how much
wall-clock that converts to depends on the machine's real cores.

0.5.0 is **Phase D — speed**: the evaluation was rewritten to be
allocation-free and to compute each piece type's attack union once
instead of twice (D1), and the main search's capture ordering now
uses static exchange evaluation instead of MVV-LVA (D2). D1 is
bit-exact — identical bench node counts to 0.4.0, verified by a
>10,000-position parity test on both parameter sets — and raises
the tuned bench rate on this VM from ~485k to ~800k nodes/sec
(+66%). D2 shrinks the fixed-depth tree by 13.6%. From 0.5.0 the
match gate is an **SPRT** (paired openings, pentanomial LLR,
elo0 = 0, elo1 = 15, α = β = 0.05, 200 ms/move); both batches were
stopped at 200 games at the owner's instruction and **kept on
their positive standings with the certification bound (+2.944)
not crossed**: D1 scored 55.8% (LLR +1.154) against 0.4.0, D2
scored 58.5% (W 83, D 69, L 48; LLR +2.771) against the D1 build.
Point estimates from those samples are roughly +40 and +60 Elo per
link (± ~25–30), compounding to roughly +100 Elo over 0.4.0; a
direct 0.5.0 vs 0.4.0 match was not run.

0.6.0 is **Phase E — search experiments**: of four gated trials,
exactly one change ships. **E1, SEE pruning in the main search**,
is the project's first fully SPRT-certified pass: against 0.5.0 it
crossed the +2.944 bound at 100 games (64.0%, LLR +3.070, point
estimate ~+100 Elo). The other three trials were run, measured and
reverted: singular extensions (E2) were SPRT-rejected at 36.0%, the
countermove heuristic (E3) stood at 44.8% at the owner-set
125-game stop, and a retune on 2200+ games only (E4) lost its
fixed-depth gate 19.0–21.0 despite fitting its own subset better.
0.6.0 is therefore 0.5.0 + SEE pruning, parameters unchanged.

0.7.0 is **Phase F — endgame**: Syzygy tablebase support and two
endgame scale factors. With `SyzygyPath` set, positions inside
the tables get exact verdicts — WDL probes inside the main
search, DTZ-optimal play at the root — and two notorious
eval blind spots (wrong-coloured bishop + rook pawn,
opposite-coloured bishops) are scaled toward draws. This phase
was commissioned as four named additions and verified by
correctness tests rather than a match gate: tablebase positions
essentially never arise from the opening suites the gates play,
so a gate would measure noise. With no path set, 0.7.0 plays
bit-identically to 0.6.0 (startpos depth 11 = 1,420,373 nodes).

0.7.5 changes **no code at all**. It is the repair of a failed
experiment: a from-scratch retune on 9.12M all-position samples
(sharp positions included — the 0.3.0 tune saw quiet positions
only) produced a *better result-prediction loss* (0.1509 vs
0.1546) and a clearly worse engine (gate: owner-stopped at 70
games, 30.7%). The shipped parameters are instead a warm-start
polish: the same optimiser starting from the 0.7.0 parameters
with a proximal penalty on drift, which moved the average
parameter by 3.90 cp and reached val 0.1536. Gated against
0.7.0 over 200 games: **W96 D42 L62 = 58.5%, LLR +2.127** —
the certification bound was not crossed, so 0.7.5 is recorded
as an uncertified keep on the standing, the D1 precedent.
The king-pressure blind spot that motivated the retune (see
the CHANGELOG) is unchanged; fixing it is 0.8.0's brief.

## What changed (the gated batches)

### Phase A — search (0.2.0)

| batch | contents | gate (timed 200 ms vs 0.1.0) | verdict |
|---|---|---|---|
| A1 | SEE capture ordering + countermove heuristic | 21.5–18.5 (53.75%) | **REVERTED** — below the 55% bar; SEE ordering also cost ~2× per node for no tree-size win |
| A2 | null-move pruning + late-move reductions + futility pruning | 27.0–13.0 (67.5%) | **KEPT** |
| A3 | aspiration windows + check extensions + quiescence SEE ordering / delta pruning / SEE pruning | 31.0–9.0 (77.5%) | **KEPT** |

Phase A totals vs the shipped 0.1.0 (tuned): timed 200 ms **31.0–9.0
(77.5%)** over 40 games; fixed depth 6 **6.0–14.0 (30%)** over 20 —
see the one-line summary for why both are true.

### Phase B — evaluation features + quiet retune (0.3.0)

22 structural features per phase block (826 parameters total, was
782): passed pawns by rank, isolated / doubled / backward pawns, king
pawn shield, king-zone attack counts by attacker type, mobility by
piece type, rook on open / semi-open file, rook on the 7th, knight
outposts. Retrained from scratch on **quiet positions only** —
2,900,016 positions (2,753,460 train / 146,556 validation) from the
same elite corpus with the same game filter and sampling as 0.1.0,
plus the quiet filter: side to move not in check, and the move
actually played from the position was not a capture, promotion or
check.

Losses (sigmoid scale 200, MSE on game results) on the **same**
held-out quiet validation set:

| parameters | train loss | val loss |
|---|---|---|
| handcrafted baseline (untuned) | 0.153311 | 0.155334 |
| 0.1.0 tuned set (old features) | 0.149097 | 0.150919 |
| ridge least squares (new features) | 0.146278 | 0.148187 |
| **0.3.0 tuned (after LM polish)** | **0.145482** | **0.147242** |

Phase B gate vs the Phase A winner (0.2.0 search, 0.1.0 parameters),
book off: fixed depth 6 **28.0–12.0 (70.0%)** over 40 games; timed
200 ms confirmation **11.5–8.5 (57.5%)** over 20. The timed margin is
thinner than the fixed-depth margin because the richer evaluation
costs nodes per second — part of the gain is spent on speed, the rest
is real. Fitted values are sane: passed pawns rise steeply by rank
(7th rank +152 MG / +184 EG), doubled pawns −37 MG, rook on the 7th
+58/+70, pawn shield +15 MG, king-zone attacks +13…+18 MG per square.

The opening book is unchanged from 0.1.0 (408,841 positions from the
same corpus, on by default, UCI option `Book`) and was off in every
match reported here.

### Phase D — speed (0.5.0)

| batch | contents | gate (SPRT, 200 ms, paired openings) | verdict |
|---|---|---|---|
| D1 | bit-exact eval speed: allocation-free feature computation, attack unions computed once and shared, pawn-feature cache, phase folded into the PST pass, Zobrist key threaded through the search | vs 0.4.0, owner-stopped at 200 games: 55.8%, LLR +1.154 | **KEPT** on the standing (bound +2.944 not crossed) |
| D2 | SEE capture ordering in the main search: SEE ≥ 0 captures ordered by exchange value above killers; SEE < 0 captures demoted below killers; MVV-LVA removed | vs the D1 build, owner-stopped at 200 games: 58.5% (W 83 D 69 L 48), LLR +2.771 | **KEPT** on the standing (bound +2.944 not crossed) |

D2 is the solo retrial of half of the reverted A1 batch (SEE
ordering was confounded with the countermove heuristic there);
tested alone, it holds up. A first D2 run was killed at game 24 by
a build-VM restart and was rerun cleanly in full; the partial run
is preserved separately and counted for nothing.

### Phase E — search experiments (0.6.0)

| batch | contents | gate | verdict |
|---|---|---|---|
| E1 | SEE pruning in the main search: at depth ≤ 2, not in check, non-mate window, after the first move, skip any move with SEE below −100 (depth 1) / −250 (depth 2) — captures and quiets alike | vs 0.5.0, SPRT 200 ms: **accept H1 at 100 games — 64.0% (W 45 D 38 L 17), LLR +3.070** | **KEPT — SPRT-certified** |
| E2 | singular extensions: TT-move verification search at half depth with the move excluded, extend +1 when every alternative fails low by 2×depth | vs the E1 build, SPRT 200 ms: **accept H0 at 100 games — 36.0% (W 18 D 37 L 45), LLR −3.075** | **REJECTED, reverted** |
| E3 | countermove heuristic: 64×64 table keyed by the previous move's (from, to), stored on quiet beta cutoffs, ordered just below the second killer | vs the E1 build, SPRT 200 ms, owner stop at 126 games: 44.8%, LLR −1.182 | **REVERTED** (negative standing) |
| E4 | retune of all 826 parameters on quiet positions from games where both players were rated ≥ 2200 (2,900,017 positions from ~182k games) | fixed depth 6, 40 games vs the 0.3.0 parameters, both on the E1 binary: 19.0–21.0 (47.5%) | **REJECTED** (55% bar) |

E1 cuts the startpos depth-11 tree from 2,094,501 nodes (0.5.0) to
1,420,373. E2's likely failure mode — recorded as analysis, not
measurement — is that at this engine's node rate and 200 ms games
the verification searches cost more than the extensions earn. E3
also settles the Phase A mystery as far as the evidence goes: the
countermove alone is a small net negative here (two independent
runs agree: 42.5% at 60 games in a restart-killed run, 44.8% at
126), so A1's reverted 53.75% was most plausibly the countermove
dragging the separately proven SEE ordering down. E4 fit its own
subset better — validation loss 0.14227 vs the 0.3.0 parameters'
0.14590 on the 2200+ validation split — and still lost at the
board: subset loss is not playing strength. After every revert the
tree was rebuilt and verified bit-identical to the E1 build
(startpos depth 11 = 1,420,373 nodes, Threads=1, book off); the
rejected E2/E3/E4 binaries are preserved as gate records. E4's
extraction also proved the shipped book is a corpus-floor
artifact: re-merging the original 2000-floor fragments reproduces
data/book.json's 408,841 positions exactly, while the 2200+
fragments merge to a different 355,534-position book. The book is
unchanged in 0.6.0.

### Phase F — endgame (0.7.0)

| addition | contents |
|---|---|
| SyzygyPath + loader | UCI string option (`;`-separated, plus `:` on non-Windows) loading `.rtbw`/`.rtbz` directories via `shakmaty-syzygy` 0.25.3 — the release built against this engine's shakmaty 0.27; the crate's own 0.27.0 release targets shakmaty 0.29 and cannot be linked here. No tables loaded = the engine behaves exactly as 0.6.0. |
| WDL probes in the main search | At non-root nodes inside table range, an exact verdict replaces the heuristic subtree search. The probe accounts for the halfmove clock, so cursed wins and blessed losses score as what they are under the fifty-move rule. A tablebase win scores ±90,000, ply-adjusted (faster wins preferred, slower losses preferred) and deliberately below the mate band, so it never masquerades as a forced mate. The probe runs after terminal detection, so checkmate and stalemate keep their exact mate scores. |
| DTZ root play | A root position inside table range is answered directly with the tablebase's DTZ-optimal move (one depth-0 info line carrying the exact verdict). This is what converts won endings inside the fifty-move budget instead of shuffling. |
| Endgame scale factors | Wrong-coloured bishop + a single rook pawn vs a bare king: score 0 when the defending king blockades the pawn's file (it always reaches the promotion corner), halved otherwise. Opposite-coloured bishops with no other pieces: score halved. Both fire only in their exact material pattern. |

Two honest records. First, the DTZ sign bug the tests caught:
the crate's `best_move` reports the SUCCESSOR position's DTZ
from the opponent's perspective; the first 0.7.0 build read it
at face value and scored KQ-vs-K as a loss for the queen side.
`tests/tablebase_smoke.py` caught it before shipping; the value
is negated at the loader boundary and the suite pins the signs.
Second, the scale factors are hand-set and were **not**
match-gated — they are unit-tested on constructed positions
only, and are recorded here as such, unlike every Phase A–E
change. The tablebase additions carry no playing-strength
claim beyond their correctness evidence (next section); the
same loader probes 6- or 7-man sets unchanged if the path
points at them, as on TCEC machines.

## Final scoreboard (book off, colours alternated, varied openings)

| match | score |
|---|---|
| Phase A (0.2.0) vs 0.1.0, timed 200 ms, 40 games | 31.0 – 9.0 (77.5%) |
| Phase A (0.2.0) vs 0.1.0, fixed depth 6, 20 games | 6.0 – 14.0 (30.0%) |
| Phase B (0.3.0) vs Phase A, fixed depth 6, 40 games | 28.0 – 12.0 (70.0%) |
| Phase B (0.3.0) vs Phase A, timed 200 ms, 20 games | 11.5 – 8.5 (57.5%) |
| **0.3.0 vs 0.1.0, timed 200 ms, 40 games** | **34.5 – 5.5 (86.25%)** |
| 0.3.0 tuned vs 0.3.0 Baseline, timed 200 ms, 20 games | 14.5 – 5.5 (72.5%) |
| 0.4.0 (Threads=1) vs 0.3.0 sanity, timed 200 ms, 20 games | 12.0 – 8.0 (60.0%) |
| **0.4.0 (Threads=2) vs 0.3.0 gate, timed 200 ms, 40 games** | **27.0 – 13.0 (67.5%)** |
| 0.5.0-D1 vs 0.4.0, SPRT 200 ms, owner-stopped at 200 games | 55.8% (W 89 D 44 L 67), LLR +1.154 |
| 0.5.0-D2 vs 0.5.0-D1, SPRT 200 ms, owner-stopped at 200 games | 58.5% (W 83 D 69 L 48), LLR +2.771 |
| **0.6.0 (E1) vs 0.5.0, SPRT 200 ms — CERTIFIED at 100 games** | **64.0% (W 45 D 38 L 17), LLR +3.070 — accept H1** |
| 0.6.0-E2 vs E1 build, SPRT 200 ms — rejected at 100 games | 36.0% (W 18 D 37 L 45), LLR −3.075 — accept H0, reverted |
| 0.6.0-E3 vs E1 build, SPRT 200 ms, owner stop at 126 games | 44.8%, LLR −1.182 — reverted |
| 0.6.0-E4 params vs 0.3.0 params (E1 binary), fixed depth 6, 40 games | 19.0 – 21.0 (47.5%) — rejected |

Gate protocol note: a first 0.4.0 gate attempt was discarded as
invalid — the match harness stopped after 20 games (9.5–10.5)
because its opening list was exhausted; the harness was fixed and
the 40-game gate above was run cleanly in full. The same
opening-list cap truncated E4's first gate attempt at 20 games
(9.5–10.5, a partial that counted only as a warning); E4's
official gate used an extended 20-opening harness and ran the
full 40.

Caveats: 20–40-game batches are noisy (± ~25–40 Elo); directions here
are consistent across independent opening batches and protocols. The
0.5.0 links are 200-game SPRT runs stopped early at the owner's
instruction — positive standings, honestly short of certification.
0.6.0's SPRT gates ran under an owner-set 125-game stop: E1 and E2
both reached full certified verdicts inside it, E3 was stopped at
126 games on a negative standing and reverted on that standing.

## Verification (this VM)

- Perft exact: startpos depth 3 = 8,902; depth 4 = 197,281; Kiwipete
  depth 3 = 97,862.
- UCI smoke: all pass, including Rust-vs-Python evaluation agreement.
- Evaluation parity: the Rust feature computation matches the tuner's
  Python reference implementation **exactly** on 300 sampled corpus
  positions with the shipped parameters (300/300).
- Time-abort regression suite (ported from Proty V1): all pass — the
  queen-grab position returns **d7g4** at `go movetime 3000` and at
  `go depth 18 movetime 3000`; the knight-rampage hang position
  answers in ~1 s at `go movetime 1000`; a mid-search `stop` is
  answered immediately.
- Stalemate-trap suite: **6/6**.
- Speed (bench_nps.py protocol: depth 9, startpos + Kiwipete +
  middlegame, engine-reported nodes/time, measured unloaded on this
  VM; single-threaded):

  | build | tuned | baseline params |
  |---|---|---|
  | 0.1.0 | 1,331,116 nps | 1,343,627 nps |
  | 0.2.0 | 1,070,844 nps | 1,111,518 nps |
  | 0.3.0 | 400,052 nps | 440,624 nps |

  Raw node rate falls across the upgrade — Phase B's evaluation
  computes attack unions for every piece at every node — but the
  searched trees shrink far more: startpos depth 9 takes 11.75M
  nodes in 0.1.0, 474k in 0.2.0 and 409k in 0.3.0, so time-to-depth
  still improves ~8× end to end. NPS is not playing strength; the
  scoreboard above is.

  Phase D (binary `bench` subcommand, depth-11 aggregate over its
  built-in position set, this VM): 0.4.0 searches 8,202,559 nodes
  at ~485k nps (tuned). D1 searches the **identical** 8,202,559
  nodes at ~800k nps — the bit-exactness proof and the speed proof
  in one pair of numbers. D2's ordering cuts the tree to 7,087,774
  nodes (−13.6%) at ~767k nps. Eval parity for D1 was also verified
  directly: the rewritten feature computation matches the preserved
  reference implementation exactly on >10,000 positions (fixed
  FENs + random walks) under both the tuned and baseline parameter
  sets (cargo test suite, 10/10).

  Phase E (this build): the full regression suite — cargo unit
  tests 10/10, UCI smoke, stalemate traps 6/6, time-abort
  regression, SMP smoke — is green on the shipped tree, and was
  re-run after each revert. The shipped tree is bit-identical to
  the gated E1 build: startpos depth 11 (Threads=1, book off) =
  1,420,373 nodes, against 2,094,501 for 0.5.0 under the same
  conditions.

  Phase F (this build): regression green (cargo tests 14/14 —
  the 10 pre-0.7.0 tests plus 4 new scale-factor tests — UCI
  smoke, stalemate traps 6/6, time-abort, SMP smoke). Tablebase
  verification ran against the complete 3-4-5-man set: 290
  files, 939 MiB, every file's size verified against the
  Lichess mirror it was downloaded from, magic bytes checked.
  Against those tables (`tests/tablebase_smoke.py`, 8/8): the
  `SyzygyPath` option is listed; KQ-vs-K scores +90,000 for
  White to move and −90,000 for Black; KR-vs-KP +90,000; K-vs-K
  and the blockaded wrong-coloured-bishop KBP both score 0
  (drawn); a 6-piece root whose Qxd4 enters a won 5-man scores
  89,999 at depth 3 in 121 nodes (heuristic without tables:
  1,177 in 966 nodes) and plays Qxd4; and the tablebase side
  converts KQ-vs-K to checkmate in 9 plies against a searching
  defender. The tables are not in the engine zip — they are
  version-independent data, distributed once as a separate
  archive (`syzygy-5man.zip`, flat `.rtbw`/`.rtbz` files);
  unzip anywhere and point `SyzygyPath` at the folder.

## Options

`Hash` (MB, default 16 in the engine — set 64 for matches), `Book`
(default true), `Baseline` (default false; true = untuned handcrafted
parameters with the Phase B features zeroed), `Threads` (1–16,
default 1; Lazy SMP helper threads sharing one transposition table),
`SyzygyPath` (string, default empty; directory or directories
holding Syzygy `.rtbw`/`.rtbz` files — `;`-separated, plus
`:`-separated on non-Windows).
Standard UCI otherwise.

Build: `cargo build --release`. Sources, extractor, tuners and tests
live in the project tree (`tools/extract_v2.py`, `tools/train_v2.py`,
`tools/extract_quiet.py`, `tools/quiet_features.py`,
`tools/train_v3.py`, `tests/`). Full per-batch gate logs are under
`data/upgrades/` in the project tree; SPRT gate logs and PGNs are
under `sprt-runs/` (`tests/match_sprt.py` is the harness).
