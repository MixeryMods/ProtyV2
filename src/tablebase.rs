//! Syzygy tablebase support (0.7.0), via the `shakmaty-syzygy`
//! crate: WDL probes inside the main search and DTZ-optimal move
//! choice at the root.
//!
//! The tables themselves are data, not code: the UCI option
//! `SyzygyPath` points at the directory (or directories) holding
//! the `.rtbw` / `.rtbz` files — `;`-separated everywhere, plus
//! `:`-separated on non-Windows, matching Stockfish convention.
//! With no path set, or a path that yields no tables, every probe
//! returns `None` and the engine plays exactly as 0.6.0 did.
//!
//! The loaded collection lives in a global read-write lock: the
//! UCI thread replaces it on `setoption`, search threads only
//! read. Probing is gated on the position's piece count, so in
//! opening and middlegame positions the only cost is one popcount.

use std::sync::RwLock;

use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::{MaybeRounded, Tablebase, Wdl};

static TABLES: RwLock<Option<Tablebase<Chess>>> = RwLock::new(None);

fn read_tables() -> std::sync::RwLockReadGuard<'static, Option<Tablebase<Chess>>> {
    TABLES.read().unwrap_or_else(|e| e.into_inner())
}

/// Replace the loaded tables from a `SyzygyPath` option value.
/// Returns the number of table files loaded (0 clears the tables).
pub fn set_path(value: &str) -> usize {
    let mut tables = Tablebase::new();
    let mut added = 0usize;
    let mut dirs: Vec<&str> = Vec::new();
    for part in value.split(';') {
        if cfg!(windows) {
            dirs.push(part);
        } else {
            dirs.extend(part.split(':'));
        }
    }
    for dir in dirs {
        let dir = dir.trim();
        if dir.is_empty() {
            continue;
        }
        if let Ok(n) = tables.add_directory(dir) {
            added += n;
        }
    }
    let mut slot = TABLES.write().unwrap_or_else(|e| e.into_inner());
    *slot = if added > 0 { Some(tables) } else { None };
    added
}

/// Number of pieces in the largest loaded table (0 = no tables).
pub fn max_pieces() -> usize {
    read_tables()
        .as_ref()
        .map_or(0, |tables| tables.max_pieces())
}

/// Exact game-theoretic verdict of a position, from the side to
/// move: 1 = win, 0 = draw, -1 = loss, `None` = not probed (no
/// tables, too many pieces, or the position is not in the tables,
/// e.g. because it still carries castling rights).
///
/// The primary probe accounts for the halfmove clock under the
/// fifty-move rule (cursed wins / blessed losses resolve to what
/// they actually score). It needs the DTZ tables as well; when
/// those are missing the after-zeroing probe — which ignores the
/// clock — is the fallback.
pub fn probe_wdl_signum(pos: &Chess) -> Option<i32> {
    let guard = read_tables();
    let tables = guard.as_ref()?;
    if pos.board().occupied().count() as usize > tables.max_pieces() {
        return None;
    }
    match tables.probe_wdl(pos) {
        Ok(wdl) => Some(wdl.signum()),
        Err(_) => tables
            .probe_wdl_after_zeroing(pos)
            .ok()
            .map(|wdl: Wdl| (wdl as i32).signum()),
    }
}

/// The tablebase's recommended root move and the resulting DTZ
/// from the MOVER's perspective (positive when winning, negative
/// when losing, 0 when drawn). Following DTZ at the root is what
/// converts won endings inside the fifty-move budget instead of
/// shuffling.
///
/// Sign care: the crate's `best_move` reports the DTZ of the
/// SUCCESSOR position, from the opponent's perspective (a winning
/// move lands the opponent in a position with negative DTZ), so
/// the value is negated here.
pub fn best_root_move(pos: &Chess) -> Option<(Move, i32)> {
    let guard = read_tables();
    let tables = guard.as_ref()?;
    if pos.board().occupied().count() as usize > tables.max_pieces() {
        return None;
    }
    match tables.best_move(pos) {
        Ok(Some((m, dtz))) => {
            let child_dtz = match dtz {
                MaybeRounded::Rounded(d) | MaybeRounded::Precise(d) => d.0,
            };
            Some((m, -child_dtz))
        }
        _ => None,
    }
}
