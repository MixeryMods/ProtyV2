//! Negamax alpha-beta search with quiescence, transposition table,
//! iterative deepening, SEE + killer + history move ordering,
//! null-move pruning, late-move reductions, futility pruning, SEE
//! pruning at frontier depths, and
//! draw detection (fifty-move, insufficient material, repetition
//! against the game history and the current search path).
//!
//! 0.8.0 additions: razoring (depths 1-3), reverse futility pruning
//! (depths 1-6), futility pruning extended to depths 1-6, late move
//! pruning (depths 1-4), a history malus for non-cutting quiet moves
//! and history scaling between searches, full 64-bit TT keys, and a
//! transposition-table probe in quiescence (D3).
//!
//! Clock handling hardened for Proty V2 (the Proty V1 lessons, applied
//! from birth — V1's time-abort catastrophe must be unrepresentable):
//!   * movetime is honoured even when a depth limit is also given
//!     (whichever limit hits first ends the search);
//!   * an iteration aborted by the clock/stop flag is DISCARDED — the
//!     returned move always comes from the last completed iteration,
//!     never from a partial one, and abort sentinel scores are never
//!     recorded as genuine evaluations;
//!   * the previous iteration's best move is searched first at the
//!     root (explicitly, not just via the transposition table);
//!   * the clock and stop flag are polled in every recursive path,
//!     quiescence included, and every time-budgeted search carries a
//!     hard node-count failsafe so a budgeted `go` always terminates;
//!   * if not even depth 1 completes, a shallow static-evaluation pass
//!     over all root moves supplies the move — never an unevaluated
//!     generation-order move.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Bitboard, Chess, Color, EnPassantMode, Move, Position, Role, Square};

use crate::eval;
use crate::tablebase;
use crate::tuned::EvalParams;

pub const MATE: i32 = 100_000;
/// Tablebase win score (0.7.0): far above any heuristic evaluation,
/// far below the mate band, so an exact tablebase verdict dominates
/// the search without ever masquerading as a forced mate in the
/// info lines.
const TB_WIN: i32 = 90_000;
const INF: i32 = 1_000_000;
const MAX_PLY: usize = 128;
const FLAG_EXACT: u8 = 0;
const FLAG_LOWER: u8 = 1;
const FLAG_UPPER: u8 = 2;

#[derive(Clone, Default)]
pub struct Limits {
    pub depth: Option<u32>,
    pub movetime: Option<Duration>,
    pub wtime: Option<Duration>,
    pub btime: Option<Duration>,
    pub winc: Option<Duration>,
    pub binc: Option<Duration>,
    pub nodes: Option<u64>,
    pub infinite: bool,
}

pub struct InfoLine {
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub pv: Vec<Move>,
}

/// One transposition-table entry, unpacked. The shared table stores
/// it packed into a single u64 (see `pack_entry`).
#[derive(Clone)]
struct TtData {
    depth: i32,
    flag: u8,
    score: i32,
    best: Option<Move>,
}

/// Lockless transposition table shared by all Lazy SMP search
/// threads (0.4.0). Each slot is two AtomicU64s: `keys[i]` holds the
/// FULL 64-bit Zobrist key of the entry (0.8.0: previously the slot
/// held `key ^ data` and a probe verified the key only through that
/// lock; the key is now stored and compared in full) and `data[i]`
/// the packed entry. A probe reads the key, then the data, then the
/// key again; if both key reads equal the probed key the entry is
/// consistent, and a torn read simply misses. Stores write the data
/// word before the key word so a racing probe that sees the new key
/// almost always sees the new data too. No locks anywhere near the
/// search hot path.
pub struct SharedTt {
    keys: Vec<AtomicU64>,
    data: Vec<AtomicU64>,
    mask: usize,
}

impl SharedTt {
    pub fn new(hash_mb: usize) -> SharedTt {
        let bytes = hash_mb.max(1) << 20;
        // 16 bytes per slot (two u64 words).
        let mut count = (bytes / 16).max(1024);
        count = count.next_power_of_two() >> 1;
        SharedTt {
            keys: (0..count).map(|_| AtomicU64::new(0)).collect(),
            data: (0..count).map(|_| AtomicU64::new(0)).collect(),
            mask: count - 1,
        }
    }

    fn probe(&self, key: u64) -> Option<TtData> {
        if key == 0 {
            return None; // 0 is the empty-slot sentinel.
        }
        let idx = (key as usize) & self.mask;
        if self.keys[idx].load(Ordering::Relaxed) != key {
            return None;
        }
        let data = self.data[idx].load(Ordering::Relaxed);
        if self.keys[idx].load(Ordering::Relaxed) != key {
            return None; // torn read: the slot changed under us.
        }
        Some(unpack_entry(data))
    }

    fn store(&self, key: u64, data: &TtData) {
        let idx = (key as usize) & self.mask;
        let packed = pack_entry(data);
        self.data[idx].store(packed, Ordering::Relaxed);
        self.keys[idx].store(key, Ordering::Relaxed);
    }
}

/// Entry packing into one u64:
///   bits  0..21  score + 1_000_000 (scores are bounded by INF)
///   bits 21..29  depth + 128 (search depth, check extension incl.)
///   bits 29..31  bound flag
///   bits 31..54  best move (see `pack_move`), all-ones = no move
fn pack_entry(e: &TtData) -> u64 {
    let score = (e.score.clamp(-INF, INF) + INF) as u64;
    let depth = (e.depth.clamp(-128, 127) + 128) as u64;
    let mv = match &e.best {
        Some(m) => pack_move(m),
        None => MOVE_NONE,
    };
    score | (depth << 21) | ((e.flag as u64) << 29) | (mv << 31)
}

fn unpack_entry(packed: u64) -> TtData {
    let mv = (packed >> 31) & 0x7F_FFFF;
    TtData {
        score: (packed & 0x1F_FFFF) as i32 - INF,
        depth: ((packed >> 21) & 0xFF) as i32 - 128,
        flag: ((packed >> 29) & 0x3) as u8,
        best: if mv == MOVE_NONE {
            None
        } else {
            Some(unpack_move(mv))
        },
    }
}

const MOVE_NONE: u64 = 0x7F_FFFF;

/// Move packing into 23 bits: kind (2), square A (6), square B (6),
/// role (3), captured role (3, 6 = none), promotion role (3, 6 =
/// none). Square A/B are from/to for Normal and EnPassant, king/rook
/// for Castle, and unused/to for Put.
fn pack_move(m: &Move) -> u64 {
    let ri = |r: Role| eval::role_index(r) as u64;
    let opt = |o: Option<Role>| o.map_or(6, |r| ri(r));
    let (kind, a, b, role, cap, promo) = match m {
        Move::Normal {
            role,
            from,
            capture,
            to,
            promotion,
        } => (
            0,
            *from as u64,
            *to as u64,
            ri(*role),
            opt(*capture),
            opt(*promotion),
        ),
        Move::EnPassant { from, to } => (1, *from as u64, *to as u64, 0, 6, 6),
        Move::Castle { king, rook } => (2, *king as u64, *rook as u64, 5, 6, 6),
        Move::Put { role, to } => (3, 0, *to as u64, ri(*role), 6, 6),
    };
    kind | (a << 2) | (b << 8) | (role << 14) | (cap << 17) | (promo << 20)
}

fn unpack_move(bits: u64) -> Move {
    fn role(i: u64) -> Role {
        match i {
            0 => Role::Pawn,
            1 => Role::Knight,
            2 => Role::Bishop,
            3 => Role::Rook,
            4 => Role::Queen,
            _ => Role::King,
        }
    }
    fn opt(i: u64) -> Option<Role> {
        if i >= 6 {
            None
        } else {
            Some(role(i))
        }
    }
    let kind = bits & 0x3;
    let a = Square::new(((bits >> 2) & 0x3F) as u32);
    let b = Square::new(((bits >> 8) & 0x3F) as u32);
    let role_i = (bits >> 14) & 0x7;
    let cap = (bits >> 17) & 0x7;
    let promo = (bits >> 20) & 0x7;
    match kind {
        0 => Move::Normal {
            role: role(role_i),
            from: a,
            capture: opt(cap),
            to: b,
            promotion: opt(promo),
        },
        1 => Move::EnPassant { from: a, to: b },
        2 => Move::Castle { king: a, rook: b },
        _ => Move::Put { role: role(role_i), to: b },
    }
}

pub fn key_of(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

pub struct Searcher {
    tt: Arc<SharedTt>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[i32; 64]; 64],
    pub nodes: u64,
    stop_flag: Arc<AtomicBool>,
    node_limit: Option<u64>,
    stopped: bool,
    params: EvalParams,
    game_history: Vec<u64>,
    root_best: Option<Move>,
    /// Previous completed iteration's best move; searched first at
    /// the root of the next iteration.
    prev_best: Option<Move>,
    /// Lazy SMP helper diversification: this searcher's ordered root
    /// move list is rotated by this many places (0 for the main
    /// thread, which must stay deterministic).
    root_rotate: usize,
    #[cfg(not(target_arch = "wasm32"))]
    deadline: Option<Instant>,
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
}

impl Searcher {
    pub fn new(
        hash_mb: usize,
        stop_flag: Arc<AtomicBool>,
        params: EvalParams,
        game_history: Vec<u64>,
    ) -> Searcher {
        Self::with_tt(
            Arc::new(SharedTt::new(hash_mb)),
            stop_flag,
            params,
            game_history,
            0,
        )
    }

    /// A searcher on a transposition table shared with other Lazy
    /// SMP threads. `root_rotate` diversifies helper threads: their
    /// ordered root move list is rotated by that many places, so the
    /// threads explore different parts of the tree first. The main
    /// thread always passes 0.
    pub fn with_tt(
        tt: Arc<SharedTt>,
        stop_flag: Arc<AtomicBool>,
        params: EvalParams,
        game_history: Vec<u64>,
        root_rotate: usize,
    ) -> Searcher {
        Searcher {
            tt,
            killers: std::array::from_fn(|_| [None, None]),
            history: [[0; 64]; 64],
            nodes: 0,
            stop_flag,
            node_limit: None,
            stopped: false,
            params,
            game_history,
            root_best: None,
            prev_best: None,
            root_rotate,
            #[cfg(not(target_arch = "wasm32"))]
            deadline: None,
            #[cfg(not(target_arch = "wasm32"))]
            start: Instant::now(),
        }
    }

    fn check_stop(&mut self) {
        if self.stopped {
            return;
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if let Some(limit) = self.node_limit {
            if self.nodes >= limit {
                self.stopped = true;
                return;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                self.stopped = true;
            }
        }
    }

    /// Iterative deepening root search. Calls `on_info` after each
    /// completed depth. Returns the best move found, if any exists.
    ///
    /// Only COMPLETED iterations update the returned move: if the
    /// clock or the stop flag aborts an iteration, that iteration is
    /// discarded whole. If no iteration completes (the budget died
    /// inside depth 1), a shallow static pass over all root moves
    /// supplies an evaluated move instead.
    pub fn search(
        &mut self,
        pos: &Chess,
        limits: &Limits,
        on_info: &mut dyn FnMut(&InfoLine),
    ) -> Option<Move> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start = Instant::now();
            self.deadline = compute_deadline(pos, limits);
        }
        self.node_limit = limits.nodes;
        // History scaling between searches (0.8.0): the butterfly
        // history table persists across searches on this searcher,
        // but move goodness from earlier positions goes stale as the
        // game moves on. Halving every entry before each new search
        // keeps recent evidence dominant without discarding the
        // table outright. On the first search the table is all
        // zeros, so this is a no-op.
        for row in self.history.iter_mut() {
            for h in row.iter_mut() {
                *h /= 2;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(d) = self.deadline {
            // Hard node failsafe for time-budgeted searches: 4096
            // nodes per budgeted millisecond is ~7x this engine's
            // real node rate, so it never binds in normal play — it
            // exists so termination does not rest on clock polling
            // alone.
            let ms = d.saturating_duration_since(self.start).as_millis() as u64;
            let cap = ms.saturating_mul(4096).max(100_000);
            self.node_limit = Some(match self.node_limit {
                Some(n) => n.min(cap),
                None => cap,
            });
        }
        // Tablebase root shortcut (0.7.0): with tables loaded and
        // few enough pieces on the board, the tablebase's
        // DTZ-optimal move IS the best move — no search improves on
        // exact play, and following DTZ is what converts won
        // endings inside the fifty-move budget instead of
        // shuffling. One info line reports the exact verdict.
        if let Some((m, dtz)) = tablebase::best_root_move(pos) {
            let score = if dtz > 0 {
                TB_WIN
            } else if dtz < 0 {
                -TB_WIN
            } else {
                0
            };
            on_info(&InfoLine {
                depth: 0,
                score,
                nodes: 0,
                #[cfg(not(target_arch = "wasm32"))]
                elapsed_ms: self.start.elapsed().as_millis() as u64,
                #[cfg(target_arch = "wasm32")]
                elapsed_ms: 0,
                pv: vec![m.clone()],
            });
            return Some(m);
        }
        let max_depth = limits.depth.unwrap_or(64).min(MAX_PLY as u32 - 2);
        let root_key = key_of(pos);
        let mut path: Vec<u64> = Vec::with_capacity(MAX_PLY);
        let mut best: Option<Move> = None;
        self.prev_best = None;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            // Aspiration window: from depth 5 on, centre a narrow
            // window on the previous iteration's score; whenever a
            // completed search lands outside it, widen (doubling the
            // delta) and re-search. Only an attempt whose score lands
            // INSIDE its window completes the iteration — a failed
            // attempt contributes nothing but TT entries, and a
            // clock-aborted attempt discards the iteration exactly as
            // a plain aborted iteration is discarded.
            const ASP_DELTA: i32 = 25;
            let (mut alpha, mut beta) = match prev_score {
                Some(s) if depth >= 5 && s.abs() < MATE - 1000 => {
                    (s - ASP_DELTA, s + ASP_DELTA)
                }
                _ => (-INF, INF),
            };
            let mut delta = ASP_DELTA;
            let mut iter_score: Option<i32> = None;
            while iter_score.is_none() {
                self.root_best = None;
                let s = self.alpha_beta(pos, depth as i32, alpha, beta, 0, &mut path, true, root_key);
                if self.stopped {
                    break; // aborted: discard the whole iteration
                }
                if s <= alpha && alpha > -INF {
                    delta = (delta * 2).min(INF);
                    alpha = (s - delta).max(-INF);
                    continue;
                }
                if s >= beta && beta < INF {
                    delta = (delta * 2).min(INF);
                    beta = (s + delta).min(INF);
                    continue;
                }
                iter_score = Some(s);
            }
            let score = match iter_score {
                Some(s) => s,
                None => break, // discard the aborted iteration entirely
            };
            prev_score = Some(score);
            if let Some(m) = self.root_best.clone().or_else(|| self.tt_best(pos)) {
                self.prev_best = Some(m.clone());
                best = Some(m);
            }
            let pv = self.extract_pv(pos, depth as usize);
            on_info(&InfoLine {
                depth,
                score,
                nodes: self.nodes,
                #[cfg(not(target_arch = "wasm32"))]
                elapsed_ms: self.start.elapsed().as_millis() as u64,
                #[cfg(target_arch = "wasm32")]
                elapsed_ms: 0,
                pv,
            });
            // A forced mate found: no point searching deeper.
            if score.abs() > MATE - 1000 {
                break;
            }
        }
        match best {
            Some(m) => Some(m),
            None => self.shallow_best(pos),
        }
    }

    /// One-ply static evaluation of every legal move, used only when
    /// the time budget died before depth 1 could complete. Every
    /// candidate is genuinely evaluated — this never returns an
    /// arbitrary generation-order move.
    fn shallow_best(&self, pos: &Chess) -> Option<Move> {
        let mut best: Option<(i32, Move)> = None;
        for m in pos.legal_moves().iter() {
            let mut child = pos.clone();
            child.play_unchecked(m);
            let s = -eval::evaluate_stm(&child, &self.params);
            if best.as_ref().map_or(true, |(bs, _)| s > *bs) {
                best = Some((s, m.clone()));
            }
        }
        best.map(|(_, m)| m)
    }

    fn tt_best(&self, pos: &Chess) -> Option<Move> {
        self.tt.probe(key_of(pos)).and_then(|e| e.best)
    }

    fn extract_pv(&self, pos: &Chess, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current = pos.clone();
        let mut seen: Vec<u64> = Vec::new();
        for _ in 0..max_len {
            let key = key_of(&current);
            if seen.contains(&key) {
                break;
            }
            seen.push(key);
            match self.tt_best(&current) {
                Some(m) => {
                    if !current.legal_moves().contains(&m) {
                        break;
                    }
                    pv.push(m.clone());
                    let mut next = current.clone();
                    next.play_unchecked(&m);
                    current = next;
                }
                None => break,
            }
        }
        pv
    }

    /// `key` is the position's Zobrist key, computed exactly once by
    /// the caller (0.5.0: the pre-0.5.0 code recomputed `key_of` —
    /// a full board scan — up to three times per node).
    fn alpha_beta(
        &mut self,
        pos: &Chess,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        path: &mut Vec<u64>,
        allow_null: bool,
        key: u64,
    ) -> i32 {
        path.push(key);
        let score = self.alpha_beta_inner(pos, depth, alpha, beta, ply, path, allow_null, key);
        path.pop();
        score
    }

    fn alpha_beta_inner(
        &mut self,
        pos: &Chess,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        path: &mut Vec<u64>,
        allow_null: bool,
        key: u64,
    ) -> i32 {
        if ply > 0 {
            // Draws: fifty-move rule and dead material end the game;
            // repetition against game history or this search path is 0.
            if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
                return 0;
            }
            if self.game_history.contains(&key) || path[..path.len() - 1].contains(&key) {
                return 0;
            }
        }
        // Check extension: a side in check has few legal replies and
        // the position is tactical by definition, so it is searched
        // one ply deeper. Applied before the depth<=0 branch, so an
        // in-check node at depth 0 gets a real one-ply search instead
        // of dropping into quiescence.
        let in_check = pos.is_check();
        let depth = if in_check { depth + 1 } else { depth };
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply, path);
        }
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_stop();
        }
        if self.stopped {
            return 0;
        }

        let mut tt_move: Option<Move> = None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.best.clone();
            if entry.depth >= depth && ply > 0 {
                let mut s = entry.score;
                if s > MATE - 1000 {
                    s -= ply as i32;
                } else if s < -(MATE - 1000) {
                    s += ply as i32;
                }
                match entry.flag {
                    FLAG_EXACT => return s,
                    FLAG_LOWER => alpha = alpha.max(s),
                    _ => beta = beta.min(s),
                }
                if alpha >= beta {
                    return s;
                }
            }
        }

        let moves = pos.legal_moves();
        if moves.is_empty() {
            return if pos.is_check() {
                -MATE + ply as i32
            } else {
                0 // stalemate
            };
        }

        // Tablebase WDL probe (0.7.0): an exact game-theoretic
        // verdict replaces the heuristic search of this subtree.
        // It runs after the terminal check above, so checkmate and
        // stalemate keep their exact mate scores, and after the
        // fifty-move / repetition / dead-material draws at the top.
        // The probe accounts for the halfmove clock, so cursed
        // wins and blessed losses score as what they are under the
        // fifty-move rule. The ply adjustment prefers the faster
        // win (and the slower loss) among tablebase-exact lines.
        if ply > 0 {
            if let Some(signum) = tablebase::probe_wdl_signum(pos) {
                return match signum {
                    1 => TB_WIN - ply as i32,
                    -1 => -TB_WIN + ply as i32,
                    _ => 0,
                };
            }
        }

        // Static evaluation for the pruning consumers below (null-move
        // pruning at depth >= 3, futility at depth <= 2).
        let static_eval = if !in_check {
            eval::evaluate_stm(pos, &self.params)
        } else {
            0
        };

        // Razoring (0.8.0), depths 1-3: if the static eval is far
        // below alpha, a full-width search is very unlikely to pull
        // this node back to alpha. Drop straight to quiescence; if
        // even quiescence beats alpha the position was sharper than
        // the static eval suggested and the normal search runs.
        if !in_check && ply > 0 && (1..=3).contains(&depth) && alpha.abs() < MATE - 1000 {
            let margin = 200 + 150 * depth;
            if static_eval + margin <= alpha {
                let q = self.quiescence(pos, alpha, beta, ply, path);
                if self.stopped {
                    return 0;
                }
                if q <= alpha {
                    return q;
                }
            }
        }

        // Reverse futility pruning (0.8.0), depths 1-6: if the
        // static eval is above beta by a depth-scaled margin, the
        // node is so good that even a bad move sequence is unlikely
        // to drag it below beta, so the search can return early.
        // Never in check, never at the root, never near mate scores.
        if !in_check
            && ply > 0
            && (1..=6).contains(&depth)
            && beta.abs() < MATE - 1000
            && static_eval - 100 * depth >= beta
        {
            return static_eval;
        }

        // Null-move pruning: if the static eval is already at or above
        // beta, test whether the position stays above beta even after
        // passing the move and searching at a reduced depth. If so,
        // the full search is very unlikely to change the verdict.
        // Guards: never in check, never at the root, never twice in a
        // row, never with pawns-only material for the side to move
        // (zugzwang), never near mate scores.
        if allow_null
            && depth >= 3
            && !in_check
            && ply > 0
            && beta.abs() < MATE - 1000
            && static_eval >= beta
            && has_non_pawn_material(pos)
        {
            let r = 2 + depth / 4;
            if let Ok(null_pos) = pos.clone().swap_turn() {
                let null_key = key_of(&null_pos);
                let null_score = -self.alpha_beta(
                    &null_pos,
                    depth - 1 - r,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    path,
                    false,
                    null_key,
                );
                if self.stopped {
                    return 0;
                }
                if null_score >= beta {
                    return beta;
                }
            }
        }

        let mut ordered = self.order_moves(pos, &moves, tt_move.as_ref(), ply);
        if ply == 0 && self.root_rotate > 0 && ordered.len() > 1 {
            // Lazy SMP helper diversification (see with_tt).
            let rotate = self.root_rotate % ordered.len();
            ordered.rotate_left(rotate);
        }
        let mut best_score = -INF;
        let mut best_move: Option<Move> = None;
        let mut flag = FLAG_UPPER;
        let mut moves_searched = 0usize;
        // Quiet moves actually searched at this node, in order:
        // the count drives late move pruning, and the list takes
        // the history malus if a later quiet move cuts (0.8.0).
        let mut quiet_moves_searched = 0usize;
        let mut quiets_tried: Vec<(usize, usize)> = Vec::new();
        for m in ordered.drain(..) {
            let quiet = !m.is_capture() && !m.is_promotion();
            // SEE pruning (0.6.0 E1): at frontier depths, once at
            // least one move has been searched, a move whose static
            // exchange evaluation falls below a depth-scaled
            // threshold is skipped outright. Unlike futility this
            // also covers captures: D2 demoted SEE-losing captures in
            // the ordering; here the worst of them are not searched
            // at all. The threshold keeps every move that wins or
            // cleanly trades material at depth 1 and only widens the
            // cut for moves that lose serious material.
            if moves_searched > 0
                && depth <= 2
                && !in_check
                && ply > 0
                && alpha.abs() < MATE - 1000
                && beta.abs() < MATE - 1000
            {
                let threshold = if depth == 1 { -100 } else { -250 };
                if see(pos, &m) < threshold {
                    continue;
                }
            }
            // Late move pruning (0.8.0), depths 1-4: once enough
            // quiet moves have been searched at a shallow node, the
            // remaining quiet moves are skipped — this late in the
            // ordering they almost never change the verdict.
            // Captures and promotions are never pruned by it, and
            // it never runs when the node is being mated.
            if quiet
                && (1..=4).contains(&depth)
                && !in_check
                && ply > 0
                && best_score > -(MATE - 1000)
                && quiet_moves_searched >= (3 + depth * depth) as usize
            {
                continue;
            }
            // Futility pruning in the move loop (extended in
            // 0.8.0 from depths 1-2 to depths 1-6, margin still
            // 150/depth so depths 1-2 behave exactly as before):
            // once at least one move has been searched, a quiet
            // move whose static eval plus the margin still cannot
            // reach alpha is skipped — it will not change this
            // node's verdict.
            if quiet
                && moves_searched > 0
                && (1..=6).contains(&depth)
                && !in_check
                && ply > 0
                && alpha.abs() < MATE - 1000
            {
                if static_eval + 150 * depth <= alpha {
                    continue;
                }
            }
            let mut child = pos.clone();
            child.play_unchecked(&m);
            let child_key = key_of(&child);
            // Late-move reductions: late quiet moves are first probed
            // at reduced depth (full window — this search stays plain
            // alpha-beta) and re-searched at full depth only if the
            // probe beats alpha.
            let reduction = if quiet && !in_check && depth >= 3 && moves_searched >= 3 {
                if depth >= 6 && moves_searched >= 8 {
                    2
                } else {
                    1
                }
            } else {
                0
            };
            let mut score = -self.alpha_beta(
                &child,
                depth - 1 - reduction,
                -beta,
                -alpha,
                ply + 1,
                path,
                true,
                child_key,
            );
            if reduction > 0 && score > alpha && !self.stopped {
                score = -self.alpha_beta(
                    &child,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    path,
                    true,
                    child_key,
                );
            }
            if self.stopped {
                return 0;
            }
            moves_searched += 1;
            if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
                if ply == 0 {
                    self.root_best = Some(m.clone());
                }
            }
            if score > alpha {
                alpha = score;
                flag = FLAG_EXACT;
            }
            if alpha >= beta {
                flag = FLAG_LOWER;
                if !m.is_capture() && !m.is_promotion() {
                    self.store_killer(ply, &m);
                    if let (Some(from), to) = (m.from(), m.to()) {
                        let h = &mut self.history[from as usize][to as usize];
                        *h += depth * depth;
                    }
                    // History malus (0.8.0): every quiet move tried
                    // earlier at this node failed to cut — push each
                    // of them down by the same depth-squared amount
                    // the cutting move was pushed up.
                    for &(f, t) in &quiets_tried {
                        self.history[f][t] -= depth * depth;
                    }
                }
                break;
            }
            if quiet {
                quiet_moves_searched += 1;
                if let (Some(from), to) = (m.from(), m.to()) {
                    quiets_tried.push((from as usize, to as usize));
                }
            }
        }

        let mut stored = best_score;
        if stored > MATE - 1000 {
            stored += ply as i32;
        } else if stored < -(MATE - 1000) {
            stored -= ply as i32;
        }
        self.tt.store(
            key,
            &TtData {
                depth,
                flag,
                score: stored,
                best: best_move,
            },
        );
        best_score
    }

    fn quiescence(
        &mut self,
        pos: &Chess,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        path: &mut Vec<u64>,
    ) -> i32 {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_stop();
        }
        if self.stopped {
            return 0;
        }
        if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
            return 0;
        }
        // D3 (0.8.0): probe the transposition table. Entries found
        // here were stored by the main search at greater depth, so
        // their bounds are at least as trustworthy as a fresh
        // quiescence search: an exact entry, or a bound that closes
        // the window, ends the node at once, and the stored best
        // move is ordered first below. Probe-only — quiescence
        // never stores, so the table is not flooded with shallow
        // entries evicting real search results.
        let key = key_of(pos);
        let mut tt_move: Option<Move> = None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.best.clone();
            let mut s = entry.score;
            if s > MATE - 1000 {
                s -= ply as i32;
            } else if s < -(MATE - 1000) {
                s += ply as i32;
            }
            match entry.flag {
                FLAG_EXACT => return s,
                FLAG_LOWER => alpha = alpha.max(s),
                _ => beta = beta.min(s),
            }
            if alpha >= beta {
                return s;
            }
        }
        let in_check = pos.is_check();
        let stand_pat = eval::evaluate_stm(pos, &self.params);
        if !in_check {
            if stand_pat >= beta {
                return beta;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }
        let legal = pos.legal_moves();
        if legal.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        let mut scored: Vec<(i32, Move)> = legal
            .iter()
            .filter(|m| in_check || m.is_capture() || m.is_promotion())
            .map(|m| {
                // Order by the exchange value for captures (best
                // first); bare promotions get queen-ish priority.
                // D3: the transposition table's move goes first.
                let s = if Some(m) == tt_move.as_ref() {
                    100_000_000
                } else if m.is_capture() {
                    see(pos, m)
                } else {
                    900
                };
                (s, m.clone())
            })
            .collect();
        scored.sort_by_key(|(s, _)| -*s);
        for (exchange, m) in scored {
            if !in_check {
                // Delta pruning: even winning the victim outright
                // cannot lift this node to alpha.
                if !m.is_promotion() {
                    if let Some(victim) = m.capture() {
                        if stand_pat + SEE_VALUES[eval::role_index(victim)] + 200 <= alpha {
                            continue;
                        }
                    }
                }
                // SEE pruning: a capture whose exchange loses material
                // and whose exchange value alone cannot reach alpha
                // is noise in a quiet-position search — skip it.
                if m.is_capture() && exchange < 0 && stand_pat + exchange <= alpha {
                    continue;
                }
            }
            let mut child = pos.clone();
            child.play_unchecked(&m);
            path.push(key_of(&child));
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1, path);
            path.pop();
            if self.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    fn store_killer(&mut self, ply: usize, m: &Move) {
        if ply >= MAX_PLY {
            return;
        }
        let slot = &mut self.killers[ply];
        if slot[0].as_ref() != Some(m) {
            slot[1] = slot[0].clone();
            slot[0] = Some(m.clone());
        }
    }

    fn order_moves(
        &self,
        pos: &Chess,
        moves: &shakmaty::MoveList,
        tt_move: Option<&Move>,
        ply: usize,
    ) -> Vec<Move> {
        let mut scored: Vec<(i32, Move)> = moves
            .iter()
            .map(|m| {
                let s = if ply == 0 && Some(m) == self.prev_best.as_ref() {
                    // Previous completed iteration's best move is
                    // always searched first at the root.
                    40_000_000
                } else if Some(m) == tt_move {
                    30_000_000
                } else if m.is_capture() {
                    // 0.5.0 D2: SEE capture ordering. Captures that
                    // win or trade material are searched first, best
                    // exchange first; captures that lose material
                    // drop below the killers — they are usually noise.
                    // (MVV-LVA, the pre-D2 scheme, never looked past
                    // the first capture of the exchange.)
                    let exchange = see(pos, m);
                    if exchange >= 0 {
                        20_000_000 + exchange
                    } else {
                        5_000_000 + exchange
                    }
                } else if m.is_promotion() {
                    15_000_000
                } else if ply < MAX_PLY && self.killers[ply][0].as_ref() == Some(m) {
                    10_000_000
                } else if ply < MAX_PLY && self.killers[ply][1].as_ref() == Some(m) {
                    9_000_000
                } else if let (Some(from), to) = (m.from(), m.to()) {
                    self.history[from as usize][to as usize]
                } else {
                    0
                };
                (s, m.clone())
            })
            .collect();
        scored.sort_by_key(|(s, _)| -*s);
        scored.into_iter().map(|(_, m)| m).collect()
    }
}

/// Does the side to move have any non-pawn material? Null-move
/// pruning is unsafe in pawn-only endings (zugzwang), so it asks.
fn has_non_pawn_material(pos: &Chess) -> bool {
    let board = pos.board();
    let us = board.by_color(pos.turn());
    !(us & (board.knights() | board.bishops() | board.rooks() | board.queens())).is_empty()
}

/// Piece values for SEE (king effectively priceless: it can only ever
/// be the last recapturer, and the swap loop guards it specially).
const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20_000];

/// Static exchange evaluation of a capture: the net material result
/// (centipawns, from the mover's perspective) of the capture sequence
/// on the move's target square when both sides keep recapturing with
/// their least valuable remaining attacker, either side allowed to
/// stop the sequence. Non-captures return 0. Pins are ignored (the
/// standard pseudo-legal SEE).
pub fn see(pos: &Chess, m: &Move) -> i32 {
    let victim = match m.capture() {
        Some(role) => role,
        None => return 0,
    };
    let board = pos.board();
    let target = m.to();
    let from = match m.from() {
        Some(sq) => sq,
        None => return 0,
    };
    let mut occupied = board.occupied();
    // The mover leaves its origin; the victim leaves the board; the
    // mover then notionally occupies the target (it is the next
    // victim). For ordinary captures the victim sat on the target;
    // for en passant it sat beside the mover.
    occupied = occupied.without(Bitboard::from(from));
    if m.is_en_passant() {
        let cap_sq = Square::from_coords(target.file(), from.rank());
        occupied = occupied.without(Bitboard::from(cap_sq));
    } else {
        occupied = occupied.without(Bitboard::from(target));
    }
    occupied = occupied.with(Bitboard::from(target));

    let attackers_of = |side: Color, occupied: Bitboard| -> Bitboard {
        board.attacks_to(target, side, occupied) & occupied
    };
    let mut gain: [i32; 32] = [0; 32];
    gain[0] = SEE_VALUES[eval::role_index(victim)];
    let mut capturer_value = SEE_VALUES[eval::role_index(m.role())];
    let mut side = pos.turn().other();
    let mut attackers = attackers_of(side, occupied);
    let mut depth = 0usize;
    loop {
        // Least valuable remaining attacker of `side`.
        let mut lva: Option<(Square, i32)> = None;
        for role in [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ] {
            let bb = attackers & board.by_role(role);
            if !bb.is_empty() {
                lva = Some((bb.first().unwrap(), SEE_VALUES[eval::role_index(role)]));
                break;
            }
        }
        let (sq, value) = match lva {
            Some(v) => v,
            None => break,
        };
        if value >= SEE_VALUES[5] && !attackers_of(side.other(), occupied).is_empty() {
            // The king may not recapture onto a still-defended square.
            break;
        }
        depth += 1;
        if depth >= 32 {
            break;
        }
        gain[depth] = capturer_value - gain[depth - 1];
        capturer_value = value;
        occupied = occupied.without(Bitboard::from(sq));
        side = side.other();
        attackers = attackers_of(side, occupied);
    }
    // Minimax the swap list back: at each step the side to capture
    // may instead stop the sequence.
    while depth > 0 {
        gain[depth - 1] = -(-gain[depth - 1]).max(gain[depth]);
        depth -= 1;
    }
    gain[0]
}

#[cfg(test)]
mod see_tests {
    use super::*;
    use shakmaty::{fen::Fen, uci::UciMove, CastlingMode};

    fn position(fen: &str) -> Chess {
        Fen::from_ascii(fen.as_bytes())
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap()
    }

    fn find(pos: &Chess, uci: &str) -> Move {
        UciMove::from_ascii(uci.as_bytes())
            .unwrap()
            .to_move(pos)
            .unwrap()
    }

    #[test]
    fn see_free_pawn() {
        let p = position("1k6/8/8/3p4/4P3/8/8/K7 w - - 0 1");
        assert_eq!(see(&p, &find(&p, "e4d5")), 100);
    }

    #[test]
    fn see_defended_pawn_equal() {
        let p = position("1k6/8/2p5/3p4/4P3/8/8/K7 w - - 0 1");
        assert_eq!(see(&p, &find(&p, "e4d5")), 0);
    }

    #[test]
    fn see_queen_takes_defended_pawn() {
        let p = position("1k6/8/2p5/3p4/8/8/8/K2Q4 w - - 0 1");
        assert_eq!(see(&p, &find(&p, "d1d5")), -800);
    }

    #[test]
    fn see_rook_battery_xray() {
        // Rd2xd5 Rxd5 Rxd5: the second white rook is revealed once the
        // first leaves d2 — net +100 for White.
        let p = position("1k1r4/8/8/3p4/8/8/3R4/K2R4 w - - 0 1");
        assert_eq!(see(&p, &find(&p, "d2d5")), 100);
    }

    #[test]
    fn see_en_passant_free() {
        let p = position("1k6/8/8/2PpP3/8/8/8/K7 w - d6 0 1");
        assert_eq!(see(&p, &find(&p, "e5d6")), 100);
    }
}

#[cfg(test)]
mod tt_tests {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode};

    fn position(fen: &str) -> Chess {
        Fen::from_ascii(fen.as_bytes())
            .unwrap()
            .into_position::<Chess>(CastlingMode::Standard)
            .unwrap()
    }

    #[test]
    fn move_pack_round_trip_all_legal_moves() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // Kiwipete: castling both sides is on the move list.
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // En passant captures.
            "1k6/8/8/2PpP3/8/8/8/K7 w - d6 0 1",
            // Promotions, including underpromotions and a capture-promo.
            "1r6/P6k/8/8/8/8/K7/8 w - - 0 1",
        ];
        let mut checked = 0;
        for fen in fens {
            let pos = position(fen);
            for m in pos.legal_moves().iter() {
                assert_eq!(unpack_move(pack_move(m)), *m, "move {m} in {fen}");
                checked += 1;
            }
        }
        assert!(checked > 80, "only {checked} moves checked");
        // Put (Crazyhouse drop) never occurs in standard chess but
        // must still round-trip: the encoding has a kind for it.
        let put = Move::Put {
            role: Role::Queen,
            to: Square::new(27),
        };
        assert_eq!(unpack_move(pack_move(&put)), put);
    }

    #[test]
    fn entry_pack_round_trip() {
        let pos = position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let a_move = pos.legal_moves().into_iter().next().unwrap();
        for score in [-100_000, -1_000, -1, 0, 1, 999, 100_000] {
            for depth in [1, 6, 32, 127] {
                for flag in [FLAG_EXACT, FLAG_LOWER, FLAG_UPPER] {
                    for best in [None, Some(a_move.clone())] {
                        let e = TtData {
                            depth,
                            flag,
                            score,
                            best,
                        };
                        let u = unpack_entry(pack_entry(&e));
                        assert_eq!(u.score, e.score);
                        assert_eq!(u.depth, e.depth);
                        assert_eq!(u.flag, e.flag);
                        assert_eq!(u.best, e.best);
                    }
                }
            }
        }
    }

    #[test]
    fn shared_tt_store_probe() {
        let tt = SharedTt::new(1);
        let pos = position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let key = key_of(&pos);
        assert!(tt.probe(key).is_none());
        let e = TtData {
            depth: 9,
            flag: FLAG_LOWER,
            score: 42,
            best: None,
        };
        tt.store(key, &e);
        let got = tt.probe(key).expect("stored entry must probe back");
        assert_eq!(got.depth, 9);
        assert_eq!(got.flag, FLAG_LOWER);
        assert_eq!(got.score, 42);
        // A different position's key must not validate against it.
        let other = position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        assert!(tt.probe(key_of(&other)).is_none());
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn compute_deadline(pos: &Chess, limits: &Limits) -> Option<Instant> {
    if limits.infinite {
        return None;
    }
    // A movetime budget applies even when a depth limit is also
    // present: whichever limit hits first ends the search. (Anciency's
    // original rule — depth given => no deadline at all — is exactly
    // the combined time+depth defect Proty V2 must not inherit.)
    if let Some(mt) = limits.movetime {
        return Some(Instant::now() + mt);
    }
    let (time, inc) = if pos.turn() == shakmaty::Color::White {
        (limits.wtime, limits.winc)
    } else {
        (limits.btime, limits.binc)
    };
    time.map(|t| {
        let budget = t / 20 + inc.unwrap_or(Duration::ZERO) / 2;
        Instant::now() + budget.min(t / 2)
    })
}
