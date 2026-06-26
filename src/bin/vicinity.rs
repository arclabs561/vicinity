//! vicinity command-line interface: build an HNSW index from a JSONL file of
//! dense vectors and run k-NN queries against it. Behind the `cli` feature.
//!
//!   vicinity build vectors.jsonl -o index.json [--m 16 --m-max 32]
//!   vicinity search index.json --query '[0.1, 0.2, 0.3]' -k 5 [--ef 64]
//!
//! Each line of the vectors file is one record:
//!   {"id": <u32>, "vec": [<f32>, ...]}
//! Every vector must share the dimension of the first. The query is a JSON array
//! of f32. Build-once (HNSW finalizes for query speed); incremental update is a
//! separate library effort tracked in the updatable-index design.

use std::fs;
use std::io::{BufRead, BufReader};

use clap::{Parser, Subcommand};
use serde::Deserialize;
use vicinity::hnsw::HNSWIndex;

#[derive(Parser)]
#[command(
    name = "vicinity",
    about = "Build and query an HNSW nearest-neighbor index"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build an HNSW index from a JSONL file of dense vectors and save it.
    Build {
        /// JSONL file: one {"id": u32, "vec": [f32, ...]} per line.
        vectors: String,
        /// Output path for the saved index.
        #[arg(short, long)]
        out: String,
        /// HNSW M (neighbors per node).
        #[arg(long, default_value_t = 16)]
        m: usize,
        /// HNSW M_max (max neighbors at layer 0).
        #[arg(long = "m-max", default_value_t = 32)]
        m_max: usize,
    },
    /// Query a saved index for the k nearest neighbors.
    Search {
        /// Index file saved by `build`.
        index: String,
        /// Query as a JSON array of f32.
        #[arg(short, long)]
        query: String,
        /// Number of neighbors to return.
        #[arg(short, default_value_t = 10)]
        k: usize,
        /// Search-time ef (higher = more accurate, slower).
        #[arg(long, default_value_t = 64)]
        ef: usize,
    },
}

#[derive(Deserialize)]
struct VecLine {
    id: u32,
    vec: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Build {
            vectors,
            out,
            m,
            m_max,
        } => {
            let file = BufReader::new(fs::File::open(&vectors)?);
            let mut docs: Vec<VecLine> = Vec::new();
            for line in file.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                docs.push(serde_json::from_str(&line)?);
            }
            let dim = docs.first().ok_or("no vectors in input")?.vec.len();
            let n = docs.len();
            let mut index = HNSWIndex::new(dim, m, m_max)?;
            for d in docs {
                // HNSW cosine distance expects unit vectors; normalize on ingest.
                index.add(d.id, vicinity::distance::normalize(&d.vec))?;
            }
            index.build()?;
            index.save_to_file(&out)?;
            eprintln!("indexed {n} vectors -> {out}");
        }
        Cmd::Search {
            index,
            query,
            k,
            ef,
        } => {
            let idx = HNSWIndex::load_from_file(&index)?;
            let parsed: Vec<f32> = serde_json::from_str(&query)?;
            let q = vicinity::distance::normalize(&parsed);
            for (id, score) in idx.search(&q, k, ef)? {
                println!("{id}\t{score:.6}");
            }
        }
    }
    Ok(())
}
