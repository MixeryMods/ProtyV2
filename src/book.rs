//! The Anciency opening book, built by tools/make_book.py from the
//! Victorian corpus only: for each position in the first 20 plies of the
//! masters' games, the moves they played, weighted by frequency and by
//! the result the side playing the move eventually scored.
//!
//! Data lives in data/book.json (embedded at compile time):
//! `{ "<board-fen> <turn> <castling>": [[uci, weight], ...], ... }`
//! The key deliberately omits the en-passant square so the same position
//! reached by a different move order still matches.

use std::collections::HashMap;
use std::sync::OnceLock;

use shakmaty::{CastlingSide, Chess, Color, Move, Position};
use shakmaty::uci::UciMove;
use shakmaty::CastlingMode;

static BOOK_JSON: &str = include_str!("../data/book.json");

type BookMap = HashMap<String, Vec<(String, u32)>>;

fn book_map() -> &'static BookMap {
    static MAP: OnceLock<BookMap> = OnceLock::new();
    MAP.get_or_init(|| {
        serde_json::from_str(BOOK_JSON).unwrap_or_else(|_| {
            // A malformed book must never break the engine: play without it.
            HashMap::new()
        })
    })
}

pub fn book_size() -> usize {
    book_map().len()
}

/// FEN board part + side to move + castling rights, e.g.
/// "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq".
pub fn position_key(pos: &Chess) -> String {
    let board = pos.board();
    let mut key = String::with_capacity(80);
    for rank in (0..8u32).rev() {
        let mut empty = 0;
        for file in 0..8u32 {
            let sq = shakmaty::Square::from_coords(
                shakmaty::File::new(file),
                shakmaty::Rank::new(rank),
            );
            match board.piece_at(sq) {
                Some(piece) => {
                    if empty > 0 {
                        key.push_str(&empty.to_string());
                        empty = 0;
                    }
                    key.push(piece.char());
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            key.push_str(&empty.to_string());
        }
        if rank > 0 {
            key.push('/');
        }
    }
    key.push(' ');
    key.push(if pos.turn() == Color::White { 'w' } else { 'b' });
    key.push(' ');
    let castles = pos.castles();
    let mut rights = String::new();
    if castles.has(Color::White, CastlingSide::KingSide) {
        rights.push('K');
    }
    if castles.has(Color::White, CastlingSide::QueenSide) {
        rights.push('Q');
    }
    if castles.has(Color::Black, CastlingSide::KingSide) {
        rights.push('k');
    }
    if castles.has(Color::Black, CastlingSide::QueenSide) {
        rights.push('q');
    }
    key.push_str(if rights.is_empty() { "-" } else { &rights });
    key
}

/// Weighted-random book move for this position, or None if the position
/// is not in the masters' games. `seed` varies the choice between calls.
pub fn pick(pos: &Chess, seed: u64) -> Option<Move> {
    let entries = book_map().get(&position_key(pos))?;
    let total: u64 = entries.iter().map(|(_, w)| *w as u64).sum();
    if total == 0 {
        return None;
    }
    // splitmix64: small deterministic PRNG, no external crates needed.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let roll = (z ^ (z >> 31)) % total;
    let mut acc = 0u64;
    for (uci, weight) in entries {
        acc += *weight as u64;
        if roll < acc {
            let parsed = UciMove::from_ascii(uci.as_bytes()).ok()?;
            return parsed.to_move(pos).ok();
        }
    }
    None
}

/// UCI string form of a move (e1g1 for castling, e7e8q for promotion).
pub fn move_to_uci(m: &Move) -> String {
    m.to_uci(CastlingMode::Standard).to_string()
}
