# Fix 2 design — nonlinear king-attack table (scoped 2026-11-01)

Disease: Proty's eval undervalues enemy pressure on the king (the
lukebot autopsy: 10...Qb6? over Bxe3!, scored as fine). Current
king_attack features (10-13 per phase) are LINEAR counts of king-zone
squares attacked per piece type. Real king danger is convex in the
number/weight of attackers; a linear model cannot express it.

Change: Ethereal-style king danger.
- attack_units(side) = sum over enemy N/B/R/Q of
  (king-zone squares attacked by that piece) * type_weight
  (weights N=2, B=2, R=3, Q=5 as the *feature definition*, not tuned)
  plus attacker-count folded in via the bucket curve itself.
- Bucket attack_units into 8 ranges: 0, 1-2, 3-4, 5-6, 7-9, 10-13,
  14-19, 20+. One indicator feature per bucket (value = own minus
  enemy, same sign convention as the other 22 features).
- EXTRA 22 -> 30, BLOCK 413 -> 421, N_PARAMS 826 -> 842.
  EvalParams gains king_table_mg: [i32; 8], king_table_eg: [i32; 8].

Touch points:
- tools/quiet_features.py: compute_features returns 30 features.
- tools/train_v6.py (clone of v5 warm-start): EXTRA=30, warm start =
  theta_v3 scattered into the 842 layout with the 16 table params = 0
  (new capacity inert at start: candidate == champion exactly before
  tuning moves it). emit_inc gains the two table fields.
- src/tuned.rs: EvalParams + HANDCRAFTED defaults (zeros).
- src/eval.rs: both feature call sites (~L92, ~L503) compute attack
  units + bucket indicator identically to the featurizer.
- Featurize v4 bins again (30-feature ftr files), rebuild block
  caches, warm-start tune, parity check (engine features vs
  featurizer on >=10k positions, both param sets), cargo tests,
  sanity check, SPRT 200 games vs 0.7.0. Keep only on a certified
  pass (standing rule).

Fallback if Fix 2 also fails: accept 0.7.0 as the engine, move to the
0.8.0 search-constant tuning program (user soft-scheduled it), and
revisit eval with the data-diet re-extraction (ply-capped sampling).
