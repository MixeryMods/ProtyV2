//! Static evaluation: material + piece-square tables, tapered between
//! midgame and endgame by remaining non-pawn material, plus the bishop
//! pair. All numbers come from `tuned::EvalParams` — the trained set by
//! default, the handcrafted baseline when the UCI option asks for it.
//!
//! Scores are centipawns from White's perspective; `evaluate_stm`
//! converts to the side to move for negamax search. Two hand-set
//! endgame scale factors (0.7.0) adjust the final score in the
//! narrow patterns documented at `apply_endgame_scale`.

use shakmaty::{Bitboard, Chess, Color, Position, Role, Square};

use crate::tuned::EvalParams;

pub fn role_index(role: Role) -> usize {
    match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

pub fn evaluate(pos: &Chess, params: &EvalParams) -> i32 {
    apply_endgame_scale(pos, evaluate_base(pos, params))
}

/// The tuned evaluation proper: tapered material + PSTs + bishop
/// pair + the 22 structural features, with no scale factors. The
/// parity tests pin THIS computation against the reference
/// implementation, exactly as they pinned `evaluate` before 0.7.0.
fn evaluate_base(pos: &Chess, params: &EvalParams) -> i32 {
    let board = pos.board();
    let (mut mg, mut eg) = (0i32, 0i32);
    let (mut bishops_w, mut bishops_b) = (0i32, 0i32);
    // Game phase 0..24 from non-pawn material (N=1, B=1, R=2, Q=4 per
    // side), counted in the same pass as the PST sum (0.5.0: this was
    // a second full board iteration before).
    let mut phase_total = 0i32;

    for (sq, piece) in board.iter() {
        let ri = role_index(piece.role);
        let s = sq as usize;
        // PSTs are stored rank-8-first: White indexes sq^56, Black sq.
        let (idx, sign) = if piece.color == Color::White {
            (s ^ 56, 1)
        } else {
            (s, -1)
        };
        mg += sign * (params.value_mg[ri] + params.pst_mg[ri][idx]);
        eg += sign * (params.value_eg[ri] + params.pst_eg[ri][idx]);
        phase_total += match piece.role {
            Role::Knight | Role::Bishop => 1,
            Role::Rook => 2,
            Role::Queen => 4,
            _ => 0,
        };
        if piece.role == Role::Bishop {
            if piece.color == Color::White {
                bishops_w += 1;
            } else {
                bishops_b += 1;
            }
        }
    }
    let phase = phase_total.min(24);
    if bishops_w >= 2 {
        mg += params.bishop_pair_mg;
        eg += params.bishop_pair_eg;
    }
    if bishops_b >= 2 {
        mg -= params.bishop_pair_mg;
        eg -= params.bishop_pair_eg;
    }

    let x = extra_features(pos);
    for i in 0..6 {
        mg += x[i] * params.passed_mg[i];
        eg += x[i] * params.passed_eg[i];
    }
    mg += x[6] * params.isolated_mg;
    eg += x[6] * params.isolated_eg;
    mg += x[7] * params.doubled_mg;
    eg += x[7] * params.doubled_eg;
    mg += x[8] * params.backward_mg;
    eg += x[8] * params.backward_eg;
    mg += x[9] * params.shield_mg;
    eg += x[9] * params.shield_eg;
    for i in 0..4 {
        mg += x[10 + i] * params.king_attack_mg[i];
        eg += x[10 + i] * params.king_attack_eg[i];
        mg += x[14 + i] * params.mobility_mg[i];
        eg += x[14 + i] * params.mobility_eg[i];
    }
    mg += x[18] * params.rook_open_mg;
    eg += x[18] * params.rook_open_eg;
    mg += x[19] * params.rook_semi_mg;
    eg += x[19] * params.rook_semi_eg;
    mg += x[20] * params.rook_seventh_mg;
    eg += x[20] * params.rook_seventh_eg;
    mg += x[21] * params.outpost_mg;
    eg += x[21] * params.outpost_eg;

    (mg * phase + eg * (24 - phase)) / 24
}

/// Endgame scale factors (0.7.0): two patterns where the raw
/// tapered score systematically overstates the winning chances,
/// applied as a pull toward zero on the White-perspective score.
/// Each fires only in the exact material configuration it
/// describes, so ordinary positions are untouched.
fn apply_endgame_scale(pos: &Chess, score: i32) -> i32 {
    let board = pos.board();

    // Wrong-coloured bishop + a single rook pawn against a bare
    // king: the bishop can never control the promotion square.
    // If the defending king blockades the pawn's file it always
    // reaches the promotion corner in time — a dead draw. Even
    // un-blockaded the ending is far more drawish than the raw
    // material score, so it is halved and the search decides.
    let non_kings = board.occupied() ^ board.kings();
    if non_kings.count() == 2 && board.bishops().count() == 1 && board.pawns().count() == 1 {
        let bishop_sq = board.bishops().first().unwrap();
        let pawn_sq = board.pawns().first().unwrap();
        let bishop_color = board.piece_at(bishop_sq).unwrap().color;
        let pawn = board.piece_at(pawn_sq).unwrap();
        let pawn_file = (pawn_sq as i32) & 7;
        if pawn.color == bishop_color && (pawn_file == 0 || pawn_file == 7) {
            let promo = match (pawn.color, pawn_file == 0) {
                (Color::White, true) => Square::A8,
                (Color::White, false) => Square::H8,
                (Color::Black, true) => Square::A1,
                (Color::Black, false) => Square::H1,
            };
            if bishop_sq.is_dark() != promo.is_dark() {
                let defender = board.king_of(pawn.color.other()).unwrap();
                let pawn_rank = (pawn_sq as i32) >> 3;
                let def_file = (defender as i32) & 7;
                let def_rank = (defender as i32) >> 3;
                let blockaded = def_file == pawn_file
                    && match pawn.color {
                        Color::White => def_rank > pawn_rank,
                        Color::Black => def_rank < pawn_rank,
                    };
                return if blockaded { 0 } else { score / 2 };
            }
        }
    }

    // Opposite-coloured bishops with no other pieces on the board:
    // the defender's bishop covers squares the attacker's can
    // never challenge, which makes pawn majorities far less
    // valuable than the raw score claims. Halved.
    let white_bishops = board.bishops() & board.by_color(Color::White);
    let black_bishops = board.bishops() & board.by_color(Color::Black);
    if white_bishops.count() == 1
        && black_bishops.count() == 1
        && (board.knights() | board.rooks() | board.queens()).is_empty()
    {
        let w_sq = white_bishops.first().unwrap();
        let b_sq = black_bishops.first().unwrap();
        if w_sq.is_dark() != b_sq.is_dark() {
            return score / 2;
        }
    }

    score
}

/// The 22 Phase B structural features (White perspective: White's
/// value minus Black's), in the exact order and with the exact
/// definitions of tools/quiet_features.py — the tuner's reference
/// implementation. Order: passed pawns by relative rank 2..7,
/// isolated, doubled, backward, king pawn shield, king-zone attacks
/// received by the enemy king minus received by the own king per
/// attacker type (N B R Q), mobility per piece type (N B R Q),
/// rooks on open files, rooks on semi-open files, rooks on the 7th
/// rank, knight outposts.
///
/// 0.5.0 speed rewrite — the returned values are bit-exact against
/// the pre-0.5.0 implementation (guarded by the parity tests below):
/// no per-call Vec allocations, per-type attack unions are computed
/// once and shared between the king-zone and mobility features, and
/// the pawn-only features (indices 0..=8) come from a per-thread
/// cache keyed by the two pawn bitboards.
pub fn extra_features(pos: &Chess) -> [i32; 22] {
    let board = pos.board();
    let white = board.by_color(Color::White);
    let black = board.by_color(Color::Black);
    let own = [white, black];
    let pawns_bb = [board.pawns() & white, board.pawns() & black];

    let pawn_diffs = pawn_feature_diffs(pawns_bb[0], pawns_bb[1]);

    let piece_bb = [
        [
            board.knights() & white,
            board.bishops() & white,
            board.rooks() & white,
            board.queens() & white,
        ],
        [
            board.knights() & black,
            board.bishops() & black,
            board.rooks() & black,
            board.queens() & black,
        ],
    ];
    let mut unions = [[Bitboard::EMPTY; 4]; 2];
    for c in 0..2 {
        for t in 0..4 {
            let mut u = Bitboard::EMPTY;
            for sq in piece_bb[c][t] {
                u |= board.attacks_from(sq);
            }
            unions[c][t] = u;
        }
    }

    let file = |sq: Square| (sq as i32) & 7;
    let rank = |sq: Square| (sq as i32) >> 3;
    let pawn_at = |c: usize, f: i32, r: i32| -> bool {
        (0..8).contains(&f) && (0..8).contains(&r)
            && pawns_bb[c].contains(Square::new((r * 8 + f) as u32))
    };

    let mut shield = [0i32; 2];
    let mut recv = [[0i32; 4]; 2];
    let mut mobility = [[0i32; 4]; 2];
    let mut rook_open = [0i32; 2];
    let mut rook_semi = [0i32; 2];
    let mut rook_seventh = [0i32; 2];
    let mut outposts = [0i32; 2];

    let mut any_pawn_file = [false; 8];
    for c in 0..2 {
        for sq in pawns_bb[c] {
            any_pawn_file[file(sq) as usize] = true;
        }
    }

    for c in 0..2 {
        let e = 1 - c;
        let king_color = if c == 0 { Color::White } else { Color::Black };
        if let Some(ksq) = board.king_of(king_color) {
            let (kf, kr) = (file(ksq), rank(ksq));
            let step = if c == 0 { 1 } else { -1 };
            for d in [step, 2 * step] {
                for df in [-1, 0, 1] {
                    if pawn_at(c, kf + df, kr + d) {
                        shield[c] += 1;
                    }
                }
            }
            let zone = board.attacks_from(ksq) | Bitboard::from(ksq);
            for t in 0..4 {
                recv[c][t] = (zone & unions[e][t]).count() as i32;
            }
        }
        for t in 0..4 {
            mobility[c][t] = unions[c][t].without(own[c]).count() as i32;
        }
        let mut own_file_count = [0i32; 8];
        for sq in pawns_bb[c] {
            own_file_count[file(sq) as usize] += 1;
        }
        for sq in piece_bb[c][2] {
            let f = file(sq) as usize;
            if !any_pawn_file[f] {
                rook_open[c] += 1;
            } else if own_file_count[f] == 0 {
                rook_semi[c] += 1;
            }
            if rank(sq) == if c == 0 { 6 } else { 1 } {
                rook_seventh[c] += 1;
            }
        }
        let mut defended = Bitboard::EMPTY;
        for sq in pawns_bb[c] {
            defended |= board.attacks_from(sq);
        }
        for sq in piece_bb[c][0] {
            let (f, r) = (file(sq), rank(sq));
            let in_zone = if c == 0 {
                (3..=5).contains(&r)
            } else {
                (2..=4).contains(&r)
            };
            if !in_zone {
                continue;
            }
            let chase = pawns_bb[e].into_iter().any(|x| {
                (file(x) - f).abs() == 1 && if c == 0 { rank(x) > r } else { rank(x) < r }
            });
            if defended.contains(sq) && !chase {
                outposts[c] += 1;
            }
        }
    }

    let mut f = [0i32; 22];
    f[..9].copy_from_slice(&pawn_diffs);
    f[9] = shield[0] - shield[1];
    for i in 0..4 {
        f[10 + i] = recv[1][i] - recv[0][i];
        f[14 + i] = mobility[0][i] - mobility[1][i];
    }
    f[18] = rook_open[0] - rook_open[1];
    f[19] = rook_semi[0] - rook_semi[1];
    f[20] = rook_seventh[0] - rook_seventh[1];
    f[21] = outposts[0] - outposts[1];
    f
}

/// Pawn-only feature differences (White minus Black) for feature
/// indices 0..=8 — passed pawns by relative rank, isolated, doubled,
/// backward — which depend on nothing but the two pawn bitboards.
/// Served from a small direct-mapped per-thread cache: pawn
/// structures repeat constantly inside one search tree. Entries are
/// validated against both full bitboards, so a probe can never return
/// another structure's values.
fn pawn_feature_diffs(wp_bb: Bitboard, bp_bb: Bitboard) -> [i32; 9] {
    let (wp, bp) = (u64::from(wp_bb), u64::from(bp_bb));
    const BITS: u32 = 14;
    let idx = ((wp ^ bp.rotate_left(31) ^ (wp >> 17)) as usize) & ((1usize << BITS) - 1);
    let hit = PAWN_CACHE.with(|cache| {
        let cache = cache.borrow();
        let entry = &cache[idx];
        if entry.2 && entry.0 == wp && entry.1 == bp {
            Some(entry.3)
        } else {
            None
        }
    });
    if let Some(diffs) = hit {
        return diffs;
    }
    let diffs = compute_pawn_feature_diffs(wp_bb, bp_bb);
    PAWN_CACHE.with(|cache| {
        cache.borrow_mut()[idx] = (wp, bp, true, diffs);
    });
    diffs
}

thread_local! {
    /// (white pawns, black pawns, occupied, diffs) per slot.
    static PAWN_CACHE: std::cell::RefCell<Vec<(u64, u64, bool, [i32; 9])>> =
        std::cell::RefCell::new(vec![(0, 0, false, [0; 9]); 1 << 14]);
}

/// The pawn-feature computation itself, with the exact definitions
/// the pre-0.5.0 `extra_features` used inline.
fn compute_pawn_feature_diffs(wp_bb: Bitboard, bp_bb: Bitboard) -> [i32; 9] {
    let pawns_bb = [wp_bb, bp_bb];
    let mut pawn_at = [[false; 64]; 2];
    // Sized for the whole board, not the legal pawn maximum, so even
    // a crafted position with extra pawns cannot overflow the list.
    let mut pawn_lists: [[Square; 64]; 2] = [[Square::A1; 64]; 2];
    let mut pawn_counts = [0usize; 2];
    for c in 0..2 {
        for sq in pawns_bb[c] {
            pawn_at[c][sq as usize] = true;
            pawn_lists[c][pawn_counts[c]] = sq;
            pawn_counts[c] += 1;
        }
    }
    let file = |sq: Square| (sq as i32) & 7;
    let rank = |sq: Square| (sq as i32) >> 3;
    let at = |c: usize, f: i32, r: i32| -> bool {
        (0..8).contains(&f) && (0..8).contains(&r) && pawn_at[c][(r * 8 + f) as usize]
    };

    let mut passed = [[0i32; 6]; 2];
    let mut isolated = [0i32; 2];
    let mut doubled = [0i32; 2];
    let mut backward = [0i32; 2];

    for c in 0..2 {
        let e = 1 - c;
        let mine = &pawn_lists[c][..pawn_counts[c]];
        let theirs = &pawn_lists[e][..pawn_counts[e]];
        let mut file_count = [0i32; 8];
        for &sq in mine {
            file_count[file(sq) as usize] += 1;
        }
        doubled[c] = file_count.iter().map(|&n| (n - 1).max(0)).sum();
        for &sq in mine {
            let (f, r) = (file(sq), rank(sq));
            let ahead = theirs.iter().any(|&x| {
                (file(x) - f).abs() <= 1 && if c == 0 { rank(x) > r } else { rank(x) < r }
            });
            if !ahead {
                let idx = if c == 0 { r - 1 } else { 6 - r };
                if (0..6).contains(&idx) {
                    passed[c][idx as usize] += 1;
                }
            }
            let has_neighbour = (f > 0 && file_count[(f - 1) as usize] > 0)
                || (f < 7 && file_count[(f + 1) as usize] > 0);
            if !has_neighbour {
                isolated[c] += 1;
            }
            let (stop_rank, support_back) = if c == 0 { (r + 2, r - 1) } else { (r - 2, r + 1) };
            let stop_attacked = at(e, f - 1, stop_rank) || at(e, f + 1, stop_rank);
            let support = at(c, f - 1, r)
                || at(c, f + 1, r)
                || at(c, f - 1, support_back)
                || at(c, f + 1, support_back);
            if stop_attacked && !support {
                backward[c] += 1;
            }
        }
    }

    let mut diffs = [0i32; 9];
    for i in 0..6 {
        diffs[i] = passed[0][i] - passed[1][i];
    }
    diffs[6] = isolated[0] - isolated[1];
    diffs[7] = doubled[0] - doubled[1];
    diffs[8] = backward[0] - backward[1];
    diffs
}

/// Evaluation from the perspective of the side to move.
pub fn evaluate_stm(pos: &Chess, params: &EvalParams) -> i32 {
    let score = evaluate(pos, params);
    if pos.turn() == Color::White {
        score
    } else {
        -score
    }
}


#[cfg(test)]
mod reference {
    use shakmaty::{Bitboard, Chess, Color, Position, Role, Square};
    use crate::tuned::EvalParams;
    use super::role_index;

    /// Pre-0.5.0 reference copy of `evaluate`, kept for parity tests.
    pub fn evaluate_reference(pos: &Chess, params: &EvalParams) -> i32 {
        let board = pos.board();
        let (mut mg, mut eg) = (0i32, 0i32);
        let (mut bishops_w, mut bishops_b) = (0i32, 0i32);
        // Game phase 0..24 from non-pawn material (N=1, B=1, R=2, Q=4 per
        // side), counted in the same pass as the PST sum (0.5.0: this was
        // a second full board iteration before).
        let mut phase_total = 0i32;

        for (sq, piece) in board.iter() {
            let ri = role_index(piece.role);
            let s = sq as usize;
            // PSTs are stored rank-8-first: White indexes sq^56, Black sq.
            let (idx, sign) = if piece.color == Color::White {
                (s ^ 56, 1)
            } else {
                (s, -1)
            };
            mg += sign * (params.value_mg[ri] + params.pst_mg[ri][idx]);
            eg += sign * (params.value_eg[ri] + params.pst_eg[ri][idx]);
            phase_total += match piece.role {
                Role::Knight | Role::Bishop => 1,
                Role::Rook => 2,
                Role::Queen => 4,
                _ => 0,
            };
            if piece.role == Role::Bishop {
                if piece.color == Color::White {
                    bishops_w += 1;
                } else {
                    bishops_b += 1;
                }
            }
        }
        let phase = phase_total.min(24);
        if bishops_w >= 2 {
            mg += params.bishop_pair_mg;
            eg += params.bishop_pair_eg;
        }
        if bishops_b >= 2 {
            mg -= params.bishop_pair_mg;
            eg -= params.bishop_pair_eg;
        }

        let x = extra_features_reference(pos);
        for i in 0..6 {
            mg += x[i] * params.passed_mg[i];
            eg += x[i] * params.passed_eg[i];
        }
        mg += x[6] * params.isolated_mg;
        eg += x[6] * params.isolated_eg;
        mg += x[7] * params.doubled_mg;
        eg += x[7] * params.doubled_eg;
        mg += x[8] * params.backward_mg;
        eg += x[8] * params.backward_eg;
        mg += x[9] * params.shield_mg;
        eg += x[9] * params.shield_eg;
        for i in 0..4 {
            mg += x[10 + i] * params.king_attack_mg[i];
            eg += x[10 + i] * params.king_attack_eg[i];
            mg += x[14 + i] * params.mobility_mg[i];
            eg += x[14 + i] * params.mobility_eg[i];
        }
        mg += x[18] * params.rook_open_mg;
        eg += x[18] * params.rook_open_eg;
        mg += x[19] * params.rook_semi_mg;
        eg += x[19] * params.rook_semi_eg;
        mg += x[20] * params.rook_seventh_mg;
        eg += x[20] * params.rook_seventh_eg;
        mg += x[21] * params.outpost_mg;
        eg += x[21] * params.outpost_eg;

        (mg * phase + eg * (24 - phase)) / 24
    }

    /// Pre-0.5.0 reference copy of `extra_features`, kept for parity tests.
    /// The 22 Phase B structural features (White perspective: White's
    /// value minus Black's), in the exact order and with the exact
    /// definitions of tools/quiet_features.py — the tuner's reference
    /// implementation. Order: passed pawns by relative rank 2..7,
    /// isolated, doubled, backward, king pawn shield, king-zone attacks
    /// received by the enemy king minus received by the own king per
    /// attacker type (N B R Q), mobility per piece type (N B R Q),
    /// rooks on open files, rooks on semi-open files, rooks on the 7th
    /// rank, knight outposts.
    pub fn extra_features_reference(pos: &Chess) -> [i32; 22] {
        let board = pos.board();
        let mut pawns: [Vec<Square>; 2] = [Vec::new(), Vec::new()];
        let mut knights: [Vec<Square>; 2] = [Vec::new(), Vec::new()];
        let mut bishops: [Vec<Square>; 2] = [Vec::new(), Vec::new()];
        let mut rooks: [Vec<Square>; 2] = [Vec::new(), Vec::new()];
        let mut queens: [Vec<Square>; 2] = [Vec::new(), Vec::new()];
        let mut kings: [Option<Square>; 2] = [None, None];
        let mut pawn_at = [[false; 64]; 2];
        for (sq, piece) in board.iter() {
            let c = if piece.color == Color::White { 0 } else { 1 };
            match piece.role {
                Role::Pawn => {
                    pawns[c].push(sq);
                    pawn_at[c][sq as usize] = true;
                }
                Role::Knight => knights[c].push(sq),
                Role::Bishop => bishops[c].push(sq),
                Role::Rook => rooks[c].push(sq),
                Role::Queen => queens[c].push(sq),
                Role::King => kings[c] = Some(sq),
            }
        }
        let file = |sq: Square| (sq as i32) & 7;
        let rank = |sq: Square| (sq as i32) >> 3;
        let at = |arr: &[[bool; 64]; 2], c: usize, f: i32, r: i32| -> bool {
            (0..8).contains(&f) && (0..8).contains(&r) && arr[c][(r * 8 + f) as usize]
        };

        let mut passed = [[0i32; 6]; 2];
        let mut isolated = [0i32; 2];
        let mut doubled = [0i32; 2];
        let mut backward = [0i32; 2];
        let mut shield = [0i32; 2];
        let mut recv = [[0i32; 4]; 2];
        let mut mobility = [[0i32; 4]; 2];
        let mut rook_open = [0i32; 2];
        let mut rook_semi = [0i32; 2];
        let mut rook_seventh = [0i32; 2];
        let mut outposts = [0i32; 2];

        let mut any_pawn_file = [false; 8];
        for cc in 0..2 {
            for &sq in &pawns[cc] {
                any_pawn_file[file(sq) as usize] = true;
            }
        }

        for c in 0..2 {
            let e = 1 - c;
            let mut file_count = [0i32; 8];
            for &sq in &pawns[c] {
                file_count[file(sq) as usize] += 1;
            }
            doubled[c] = file_count.iter().map(|&n| (n - 1).max(0)).sum();
            for &sq in &pawns[c] {
                let (f, r) = (file(sq), rank(sq));
                let ahead = pawns[e].iter().any(|&x| {
                    (file(x) - f).abs() <= 1 && if c == 0 { rank(x) > r } else { rank(x) < r }
                });
                if !ahead {
                    let idx = if c == 0 { r - 1 } else { 6 - r };
                    if (0..6).contains(&idx) {
                        passed[c][idx as usize] += 1;
                    }
                }
                let has_neighbour = (f > 0 && file_count[(f - 1) as usize] > 0)
                    || (f < 7 && file_count[(f + 1) as usize] > 0);
                if !has_neighbour {
                    isolated[c] += 1;
                }
                let (stop_rank, support_back) = if c == 0 { (r + 2, r - 1) } else { (r - 2, r + 1) };
                let stop_attacked =
                    at(&pawn_at, e, f - 1, stop_rank) || at(&pawn_at, e, f + 1, stop_rank);
                let support = at(&pawn_at, c, f - 1, r)
                    || at(&pawn_at, c, f + 1, r)
                    || at(&pawn_at, c, f - 1, support_back)
                    || at(&pawn_at, c, f + 1, support_back);
                if stop_attacked && !support {
                    backward[c] += 1;
                }
            }
            if let Some(ksq) = kings[c] {
                let (kf, kr) = (file(ksq), rank(ksq));
                let step = if c == 0 { 1 } else { -1 };
                for d in [step, 2 * step] {
                    for df in [-1, 0, 1] {
                        if at(&pawn_at, c, kf + df, kr + d) {
                            shield[c] += 1;
                        }
                    }
                }
                let zone = board.attacks_from(ksq) | Bitboard::from(ksq);
                let enemy = [&knights[e], &bishops[e], &rooks[e], &queens[e]];
                for t in 0..4 {
                    let mut union = Bitboard::EMPTY;
                    for &sq in enemy[t] {
                        union |= board.attacks_from(sq);
                    }
                    recv[c][t] = (zone & union).count() as i32;
                }
            }
            let own_color = if c == 0 { Color::White } else { Color::Black };
            let own = board.by_color(own_color);
            let mine = [&knights[c], &bishops[c], &rooks[c], &queens[c]];
            for t in 0..4 {
                let mut union = Bitboard::EMPTY;
                for &sq in mine[t] {
                    union |= board.attacks_from(sq);
                }
                mobility[c][t] = union.without(own).count() as i32;
            }
            for &sq in &rooks[c] {
                let f = file(sq) as usize;
                if !any_pawn_file[f] {
                    rook_open[c] += 1;
                } else if file_count[f] == 0 {
                    rook_semi[c] += 1;
                }
                if rank(sq) == if c == 0 { 6 } else { 1 } {
                    rook_seventh[c] += 1;
                }
            }
            let mut defended = Bitboard::EMPTY;
            for &sq in &pawns[c] {
                defended |= board.attacks_from(sq);
            }
            for &sq in &knights[c] {
                let (f, r) = (file(sq), rank(sq));
                let in_zone = if c == 0 {
                    (3..=5).contains(&r)
                } else {
                    (2..=4).contains(&r)
                };
                if !in_zone {
                    continue;
                }
                let chase = pawns[e].iter().any(|&x| {
                    (file(x) - f).abs() == 1 && if c == 0 { rank(x) > r } else { rank(x) < r }
                });
                if defended.contains(sq) && !chase {
                    outposts[c] += 1;
                }
            }
        }

        let mut f = [0i32; 22];
        for i in 0..6 {
            f[i] = passed[0][i] - passed[1][i];
        }
        f[6] = isolated[0] - isolated[1];
        f[7] = doubled[0] - doubled[1];
        f[8] = backward[0] - backward[1];
        f[9] = shield[0] - shield[1];
        for i in 0..4 {
            f[10 + i] = recv[1][i] - recv[0][i];
            f[14 + i] = mobility[0][i] - mobility[1][i];
        }
        f[18] = rook_open[0] - rook_open[1];
        f[19] = rook_semi[0] - rook_semi[1];
        f[20] = rook_seventh[0] - rook_seventh[1];
        f[21] = outposts[0] - outposts[1];
        f
    }
}

#[cfg(test)]
mod parity_tests {
    //! 0.5.0 bit-exactness guard: the rewritten `extra_features` and
    //! `evaluate` must return identical values to the pre-0.5.0
    //! reference implementations (kept in `super::reference`) on every
    //! position — fixed FENs covering the feature edge cases plus
    //! deterministic pseudo-random game walks.

    use super::reference::{evaluate_reference, extra_features_reference};
    use super::{evaluate, evaluate_base, extra_features};
    use crate::tuned::{BASELINE, TRAINED};
    use shakmaty::{fen::Fen, Chess, CastlingMode, Move, Position};

    fn position(fen: &str) -> Chess {
        Fen::from_ascii(fen.as_bytes())
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap()
    }

    const FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 8",
        "8/pp3ppp/8/8/8/8/PP3PPP/4K2k w - - 0 1",
        "1r6/P6k/8/8/8/8/K7/8 w - - 0 1",
        "8/2P5/8/8/8/1k6/8/4K3 w - - 0 1",
        "4k3/8/8/3p4/3P4/3K4/8/8 w - - 0 1",
        "4k3/8/2p1p3/2pPp3/2P1P3/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "qq2k2q/8/8/8/8/8/8/QQ2K2Q w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pp1p1ppp/2p1pn2/8/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 5",
        "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1",
        "2rr1k2/pp1bpp1p/2n1p2p/q2P4/2B1P3/2N2N2/PP3PPP/2RQ1RK1 w - - 0 1",
        "r2q1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        "6k1/8/6PP/8/8/8/6pp/6K1 w - - 0 1",
        "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
        "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",
        "b3k2b/8/8/8/8/8/8/B3K2B w - - 0 1",
        "n3k2n/8/8/8/8/8/8/N3K2N w - - 0 1",
        "4k3/8/8/2N5/8/8/8/4K3 w - - 0 1",
        "4k3/3p4/8/4N3/8/8/8/4K3 w - - 0 1",
        "4k3/8/3p4/4N3/8/8/8/4K3 w - - 0 1",
        "1k6/8/8/2PpP3/8/8/8/K7 w - d6 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
    ];

    fn check(pos: &Chess) {
        assert_eq!(
            extra_features(pos),
            extra_features_reference(pos),
            "feature mismatch at {}",
            shakmaty::fen::Fen::from_position(pos.clone(), shakmaty::EnPassantMode::Legal)
        );
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(
                evaluate_base(pos, params),
                evaluate_reference(pos, params),
                "eval mismatch at {}",
                shakmaty::fen::Fen::from_position(pos.clone(), shakmaty::EnPassantMode::Legal)
            );
        }
    }

    #[test]
    fn parity_on_fixed_fens() {
        for fen in FENS {
            check(&position(fen));
        }
    }

    #[test]
    fn parity_on_random_walks() {
        // Deterministic LCG; walks start from several seeds positions
        // so pawn structures, promotions and bare endgames all occur.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/pp3ppp/8/8/8/8/PP3PPP/4K2k w - - 0 1",
            "1r6/P6k/8/8/8/8/K7/8 w - - 0 1",
        ];
        let mut checked = 0usize;
        for start in starts {
            for _ in 0..60 {
                let mut pos = position(start);
                for _ in 0..110 {
                    check(&pos);
                    checked += 1;
                    let moves: Vec<Move> = pos.legal_moves().iter().cloned().collect();
                    if moves.is_empty() {
                        break;
                    }
                    let m = moves[next() % moves.len()].clone();
                    pos.play_unchecked(&m);
                }
            }
        }
        assert!(checked > 10_000, "only {checked} positions checked");
    }

    #[test]
    fn scale_wrong_colour_bishop_blockaded_is_drawn() {
        // Dark-squared bishop c1, rook pawn a3, black king
        // blockading on the a-file: the bishop never controls a8
        // (a light square) and the king reaches the corner.
        let pos = position("8/8/8/k7/8/P7/1K6/2B5 w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert!(evaluate_base(&pos, params) > 100);
            assert_eq!(evaluate(&pos, params), 0);
        }
        // Same pattern with colours flipped: light-squared bishop
        // b7, black rook pawn a4, white king blockading on a2.
        let pos = position("8/1b6/8/8/p7/8/K3k3/8 b - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(evaluate(&pos, params), 0);
        }
    }

    #[test]
    fn scale_wrong_colour_bishop_unblockaded_is_halved() {
        // Defending king far away on h8: still the wrong-coloured
        // bishop pattern, but no blockade — the score is halved,
        // not zeroed.
        let pos = position("7k/8/8/8/8/P7/1K6/2B5 w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(evaluate(&pos, params), evaluate_base(&pos, params) / 2);
        }
    }

    #[test]
    fn scale_correct_colour_bishop_is_untouched() {
        // Bishop b7 is light-squared and DOES control a8: not the
        // wrong-colour pattern, even with the black king on the
        // a-file. No scaling.
        let pos = position("8/1B6/8/k7/8/P7/1K6/8 w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(evaluate(&pos, params), evaluate_base(&pos, params));
        }
    }

    #[test]
    fn scale_opposite_coloured_bishops_is_halved() {
        // Bf2 (dark) vs Bf7 (light), only pawns besides: halved.
        let pos = position("4k3/5b2/8/3P4/4P3/8/5B2/4K3 w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert!(evaluate_base(&pos, params).abs() > 50);
            assert_eq!(evaluate(&pos, params), evaluate_base(&pos, params) / 2);
        }
        // Same-coloured bishops (Bc5 is dark, like Bf2): untouched.
        let pos = position("4k3/8/8/2bP4/4P3/8/5B2/4K3 w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(evaluate(&pos, params), evaluate_base(&pos, params));
        }
        // A rook on the board breaks the pure-bishop pattern.
        let pos = position("4k3/5b2/8/3P4/4P3/8/5B2/4K2R w - - 0 1");
        for params in [&TRAINED, &BASELINE] {
            assert_eq!(evaluate(&pos, params), evaluate_base(&pos, params));
        }
    }
}
