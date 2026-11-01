//! WebAssembly bindings: the same engine, exposed to JavaScript.
//!
//! API:
//!   const engine = new Anciency();
//!   engine.set_fen("rnbqkbnr/... w KQkq - 0 1");  // or set_startpos()
//!   engine.legal_moves();        // "e2e4 d2d4 ..." (space separated UCI)
//!   engine.apply_uci("e2e4");    // play a move, returns success
//!   engine.best_move(10, 0);      // best move at depth 10 (UCI string)
//!   engine.evaluate();           // centipawns, White perspective
//!   engine.perft(3);             // legal-move tree node count
//!   engine.fen();                // current position as FEN
//!
//! Time-based search is not available in this build (std::time is not
//! implemented on wasm32-unknown-unknown), so `best_move`'s second
//! argument is accepted for API symmetry and depth governs the search.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use wasm_bindgen::prelude::*;

use crate::book;
use crate::eval;
use crate::search::{key_of, Limits, Searcher};
use crate::tuned::TRAINED;

#[wasm_bindgen]
pub struct Anciency {
    pos: Chess,
    history: Vec<u64>,
    fullmove: u32,
}

#[wasm_bindgen]
impl Anciency {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Anciency {
        let pos = Chess::default();
        Anciency {
            history: vec![key_of(&pos)],
            pos,
            fullmove: 1,
        }
    }

    pub fn set_startpos(&mut self) {
        self.pos = Chess::default();
        self.history = vec![key_of(&self.pos)];
        self.fullmove = 1;
    }

    /// Set the position from a FEN string. Returns false on a bad FEN.
    pub fn set_fen(&mut self, fen: &str) -> bool {
        let parsed = Fen::from_ascii(fen.as_bytes())
            .ok()
            .and_then(|f| f.into_position::<Chess>(CastlingMode::Standard).ok());
        match parsed {
            Some(pos) => {
                self.fullmove = fen
                    .split_whitespace()
                    .nth(5)
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(1);
                self.pos = pos;
                self.history = vec![key_of(&self.pos)];
                true
            }
            None => false,
        }
    }

    /// Play a UCI move (e.g. "e2e4", "e7e8q"). Returns false if illegal.
    pub fn apply_uci(&mut self, uci: &str) -> bool {
        let parsed = UciMove::from_ascii(uci.as_bytes())
            .ok()
            .and_then(|u| u.to_move(&self.pos).ok());
        match parsed {
            Some(m) => {
                if self.pos.turn() == shakmaty::Color::Black {
                    self.fullmove += 1;
                }
                self.pos.play_unchecked(&m);
                self.history.push(key_of(&self.pos));
                true
            }
            None => false,
        }
    }

    /// All legal moves as a space-separated string of UCI moves.
    pub fn legal_moves(&self) -> String {
        self.pos
            .legal_moves()
            .iter()
            .map(book::move_to_uci)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Best move (UCI string, "" if the game is over). Consults the
    /// Victorian opening book first, then searches to `depth`.
    pub fn best_move(&mut self, depth: u32, _movetime_ms: u32) -> String {
        if self.history.len() <= 21 {
            let seed = key_of(&self.pos) ^ (self.history.len() as u64).wrapping_mul(0x2545_F491);
            if let Some(m) = book::pick(&self.pos, seed) {
                self.apply_uci(&book::move_to_uci(&m));
                return book::move_to_uci(&m);
            }
        }
        let limits = Limits {
            depth: Some(depth.max(1)),
            ..Default::default()
        };
        let history_before: Vec<u64> =
            self.history[..self.history.len().saturating_sub(1)].to_vec();
        let mut searcher = Searcher::new(16, Arc::new(AtomicBool::new(false)), TRAINED, history_before);
        let best = searcher.search(&self.pos, &limits, &mut |_| {});
        match best {
            Some(m) => {
                let uci = book::move_to_uci(&m);
                self.apply_uci(&uci);
                uci
            }
            None => String::new(),
        }
    }

    /// Static evaluation in centipawns, from White's perspective.
    pub fn evaluate(&self) -> i32 {
        eval::evaluate(&self.pos, &TRAINED)
    }

    /// Legal-move tree size to `depth` (perft) — 8902 at depth 3 from
    /// the start position.
    pub fn perft(&self, depth: u32) -> u64 {
        shakmaty::perft(&self.pos, depth)
    }

    /// Current position as a FEN string (book key + ep, clocks).
    pub fn fen(&self) -> String {
        let ep = match self.pos.ep_square(shakmaty::EnPassantMode::Legal) {
            Some(sq) => sq.to_string(),
            None => "-".to_string(),
        };
        format!(
            "{} {} {} {}",
            book::position_key(&self.pos),
            ep,
            self.pos.halfmoves(),
            self.fullmove
        )
    }
}
