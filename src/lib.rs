//! Proty V2 — a chess engine whose handcrafted evaluation is
//! Texel-tuned on game results from the modern elite corpus
//! (titled players, CM/WCM and up). Forked from Anciency's codebase
//! (same eval shape, same search skeleton, hardened clock handling);
//! Anciency itself is a separate, untouched project.
//! Library root: evaluation, search, opening book, UCI and (on wasm32)
//! the web bindings live in the modules below.

pub mod book;
pub mod eval;
pub mod search;
pub mod tablebase;
pub mod tuned;
pub mod uci;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
