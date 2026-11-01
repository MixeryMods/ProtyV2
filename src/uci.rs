//! UCI protocol front-end for the native binary.
//!
//! Supports: uci, isready, ucinewgame, setoption (Hash, Book, Baseline,
//! Threads, SyzygyPath),
//! position (startpos / fen, with moves), go (depth, movetime, wtime /
//! btime / winc / binc, nodes, infinite, perft), stop, quit, plus the
//! debug commands `perft N`, `eval` and `d` (print position).
//! Searches run on a worker thread so `stop` and `quit` stay responsive.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};

use crate::book;
use crate::eval;
use crate::search::{key_of, InfoLine, Limits, Searcher, SharedTt, MATE};
use crate::tuned::{BASELINE, TRAINED};

static STDOUT_LOCK: Mutex<()> = Mutex::new(());

fn out(line: &str) {
    let _guard = STDOUT_LOCK.lock().unwrap();
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let _ = writeln!(handle, "{line}");
    let _ = handle.flush();
}

#[derive(Clone, Copy)]
struct Options {
    hash_mb: usize,
    book: bool,
    baseline: bool,
    threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            hash_mb: 16,
            book: true,
            baseline: false,
            threads: 1,
        }
    }
}

struct Running {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

pub fn run() {
    let stdin = std::io::stdin();
    let mut pos = Chess::default();
    let mut history: Vec<u64> = vec![key_of(&pos)];
    let mut options = Options::default();
    let mut running: Option<Running> = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "uci" => {
                out("id name Proty V2");
                out("id author 216Byte (Muse)");
                out("option name Hash type spin default 16 min 1 max 1024");
                out("option name Book type check default true");
                out("option name Baseline type check default false");
                out("option name Threads type spin default 1 min 1 max 16");
                out("option name SyzygyPath type string default <empty>");
                out("uciok");
            }
            "isready" => out("readyok"),
            "ucinewgame" => {
                stop_search(&mut running);
                pos = Chess::default();
                history = vec![key_of(&pos)];
            }
            "setoption" => apply_setoption(&tokens, &mut options),
            "position" => {
                stop_search(&mut running);
                if let Some((p, h)) = parse_position(&tokens) {
                    pos = p;
                    history = h;
                }
            }
            "go" => {
                stop_search(&mut running);
                running = start_search(&tokens, &pos, &history, options);
            }
            "stop" => stop_search(&mut running),
            "perft" => {
                let depth: u32 = tokens.get(1).and_then(|t| t.parse().ok()).unwrap_or(3);
                let n = shakmaty::perft(&pos, depth);
                out(&format!("perft {depth} = {n}"));
            }
            "eval" => {
                let params = if options.baseline { BASELINE } else { TRAINED };
                out(&format!("eval {} cp (white perspective)", eval::evaluate(&pos, &params)));
            }
            "d" => {
                out(&format!("bookkey {}", book::position_key(&pos)));
                out(&format!(
                    "book entries in total: {}",
                    crate::book::book_size()
                ));
            }
            "quit" => {
                stop_search(&mut running);
                break;
            }
            _ => {} // unknown commands are ignored, per UCI custom
        }
    }
}

fn stop_search(running: &mut Option<Running>) {
    if let Some(r) = running.take() {
        r.stop.store(true, Ordering::Relaxed);
        let _ = r.handle.join();
    }
}

fn apply_setoption(tokens: &[&str], options: &mut Options) {
    // setoption name <name...> [value <value...>]
    let mut name = String::new();
    let mut value = String::new();
    let mut in_value = false;
    for tok in &tokens[1..] {
        if *tok == "value" {
            in_value = true;
        } else if in_value {
            if !value.is_empty() {
                value.push(' ');
            }
            value.push_str(tok);
        } else if *tok != "name" {
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(tok);
        }
    }
    match name.to_lowercase().as_str() {
        "hash" => {
            if let Ok(v) = value.parse::<usize>() {
                options.hash_mb = v.clamp(1, 1024);
            }
        }
        "book" => options.book = value.eq_ignore_ascii_case("true"),
        "baseline" => options.baseline = value.eq_ignore_ascii_case("true"),
        "threads" => {
            if let Ok(v) = value.parse::<usize>() {
                options.threads = v.clamp(1, 16);
            }
        }
        "syzygypath" => {
            crate::tablebase::set_path(&value);
        }
        _ => {}
    }
}

fn parse_position(tokens: &[&str]) -> Option<(Chess, Vec<u64>)> {
    let mut idx = 1;
    let mut pos = if tokens.get(idx) == Some(&"startpos") {
        idx += 1;
        Chess::default()
    } else if tokens.get(idx) == Some(&"fen") {
        idx += 1;
        let end = tokens[idx..]
            .iter()
            .position(|t| *t == "moves")
            .map(|p| idx + p)
            .unwrap_or(tokens.len());
        let fen_str = tokens[idx..end].join(" ");
        idx = end;
        Fen::from_ascii(fen_str.as_bytes())
            .ok()?
            .into_position::<Chess>(CastlingMode::Standard)
            .ok()?
    } else {
        return None;
    };
    let mut history = vec![key_of(&pos)];
    if tokens.get(idx) == Some(&"moves") {
        for tok in &tokens[idx + 1..] {
            let uci = UciMove::from_ascii(tok.as_bytes()).ok()?;
            let m = uci.to_move(&pos).ok()?;
            pos.play_unchecked(&m);
            history.push(key_of(&pos));
        }
    }
    Some((pos, history))
}

fn parse_limits(tokens: &[&str]) -> Limits {
    let mut limits = Limits::default();
    let mut i = 1;
    while i < tokens.len() {
        let next = tokens.get(i + 1).and_then(|t| t.parse::<u64>().ok());
        match tokens[i] {
            "depth" => limits.depth = next.map(|v| v as u32),
            "movetime" => limits.movetime = next.map(Duration::from_millis),
            "wtime" => limits.wtime = next.map(Duration::from_millis),
            "btime" => limits.btime = next.map(Duration::from_millis),
            "winc" => limits.winc = next.map(Duration::from_millis),
            "binc" => limits.binc = next.map(Duration::from_millis),
            "nodes" => limits.nodes = next,
            "infinite" => {
                limits.infinite = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        i += 2;
    }
    limits
}

fn start_search(
    tokens: &[&str],
    pos: &Chess,
    history: &[u64],
    options: Options,
) -> Option<Running> {
    // `go perft N` is answered synchronously and starts no search.
    if let Some(p) = tokens.iter().position(|t| *t == "perft") {
        if let Some(d) = tokens.get(p + 1).and_then(|t| t.parse::<u32>().ok()) {
            let n = shakmaty::perft(pos, d);
            out(&format!("info depth {d} nodes {n} string perft"));
            out("bestmove (none)");
            return None;
        }
    }

    // Opening book: only early in the game, and only from positions the
    // masters actually reached (later positions simply miss the book).
    if options.book && history.len() <= 21 {
        let seed = key_of(pos) ^ (history.len() as u64).wrapping_mul(0x2545_F491);
        if let Some(m) = book::pick(pos, seed) {
            out(&format!("info string book move {}", book::move_to_uci(&m)));
            out(&format!("bestmove {}", book::move_to_uci(&m)));
            return None;
        }
    }

    if pos.legal_moves().is_empty() {
        out("bestmove (none)");
        return None;
    }

    let limits = parse_limits(tokens);
    let params = if options.baseline { BASELINE } else { TRAINED };
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let pos_thread = pos.clone();
    let history_thread: Vec<u64> = history[..history.len().saturating_sub(1)].to_vec();
    let handle = std::thread::spawn(move || {
        // Lazy SMP (0.4.0): with Threads > 1 the main searcher shares
        // its transposition table with helper searchers whose root
        // move orders are rotated, so they explore different parts of
        // the tree first. Only the main searcher prints info lines,
        // and the returned move is the main searcher's — produced
        // under the same completed-iteration clock rules as always.
        // When the main search ends (limit, clock, or the shared stop
        // flag), the flag is set and the helpers are joined before
        // the bestmove is printed, so `stop`/`quit` stay prompt.
        let tt = Arc::new(SharedTt::new(options.hash_mb));
        let mut helpers = Vec::new();
        for t in 1..options.threads {
            let htt = tt.clone();
            let hstop = stop_thread.clone();
            let hpos = pos_thread.clone();
            let hlimits = limits.clone();
            let hhistory = history_thread.clone();
            helpers.push(std::thread::spawn(move || {
                let mut helper = Searcher::with_tt(htt, hstop, params, hhistory, t);
                let _ = helper.search(&hpos, &hlimits, &mut |_: &InfoLine| {});
            }));
        }
        let mut searcher =
            Searcher::with_tt(tt, stop_thread.clone(), params, history_thread, 0);
        let best = searcher.search(&pos_thread, &limits, &mut |info: &InfoLine| {
            out(&format_info(info));
        });
        stop_thread.store(true, Ordering::Relaxed);
        for h in helpers {
            let _ = h.join();
        }
        match best {
            Some(m) => out(&format!("bestmove {}", book::move_to_uci(&m))),
            None => out("bestmove (none)"),
        }
    });
    Some(Running { stop, handle })
}

fn format_info(info: &InfoLine) -> String {
    let score = if info.score > MATE - 1000 {
        format!("mate {}", (MATE - info.score + 1) / 2)
    } else if info.score < -(MATE - 1000) {
        format!("mate {}", -(MATE + info.score + 1) / 2)
    } else {
        format!("cp {}", info.score)
    };
    let pv: Vec<String> = info.pv.iter().map(book::move_to_uci).collect();
    format!(
        "info depth {} score {} nodes {} time {} pv {}",
        info.depth,
        score,
        info.nodes,
        info.elapsed_ms,
        pv.join(" ")
    )
}
