//! SparseMIPS benchmark over SPV1/NBR1 smoke datasets.
//!
//! ```bash
//! uv run scripts/generate_sparse_mips_smoke_data.py data/sparse-mips/smoke
//! cargo run --example sparse_mips_benchmark --release --features sparse_mips -- \
//!   data/sparse-mips/smoke
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::Instant;

use vicinity::sparse_mips::{SparseMipsIndex, SparseMipsParams, SparseVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/sparse-mips/smoke".to_string());
    let data_dir = Path::new(&data_dir);

    if !data_dir.join("train.spv1").exists() {
        println!(
            "SparseMIPS smoke data not found at {}. Run: uv run scripts/generate_sparse_mips_smoke_data.py {}",
            data_dir.display(),
            data_dir.display()
        );
        return Ok(());
    }

    let train = load_spv1(&data_dir.join("train.spv1"))?;
    let test = load_spv1(&data_dir.join("test.spv1"))?;
    let (neighbors, gt_k) = load_nbr1(&data_dir.join("neighbors.bin"))?;
    let k = gt_k.min(10);

    let params = SparseMipsParams {
        max_degree: 24,
        ef_construction: 80,
        ef_search: 80,
        alpha: 1.2,
    };

    let build_start = Instant::now();
    let mut index = SparseMipsIndex::new(params);
    for (doc_id, vector) in train.iter().enumerate() {
        index.add(doc_id as u32, vector.clone())?;
    }
    index.build()?;
    let build_time = build_start.elapsed();

    let search_start = Instant::now();
    let mut results = Vec::with_capacity(test.len());
    for query in &test {
        results.push(index.search(query, k)?);
    }
    let search_time = search_start.elapsed();

    let recall = recall_at_k(&results, &neighbors, k);
    let qps = if search_time.as_secs_f64() > 0.0 {
        test.len() as f64 / search_time.as_secs_f64()
    } else {
        f64::INFINITY
    };

    println!("SparseMIPS smoke benchmark");
    println!("vectors={} queries={} k={}", train.len(), test.len(), k);
    println!("build_time_s={:.6}", build_time.as_secs_f64());
    println!("recall_at_{k}={recall:.4}");
    println!("qps={qps:.1}");

    Ok(())
}

fn load_spv1(path: &Path) -> Result<Vec<SparseVector>, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"SPV1" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid SPV1 magic").into());
    }

    let rows = read_u32(&mut reader)? as usize;
    let total_nnz = read_u64(&mut reader)? as usize;

    let mut offsets = Vec::with_capacity(rows + 1);
    for _ in 0..=rows {
        offsets.push(read_u64(&mut reader)? as usize);
    }
    if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(total_nnz) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid SPV1 offsets").into());
    }
    if !offsets.windows(2).all(|window| window[0] <= window[1]) {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "non-monotonic SPV1 offsets").into(),
        );
    }

    let mut indices = Vec::with_capacity(total_nnz);
    for _ in 0..total_nnz {
        indices.push(read_u32(&mut reader)?);
    }
    let mut values = Vec::with_capacity(total_nnz);
    for _ in 0..total_nnz {
        values.push(read_f32(&mut reader)?);
    }

    let mut vectors = Vec::with_capacity(rows);
    for window in offsets.windows(2) {
        let start = window[0];
        let end = window[1];
        let vector_indices = indices[start..end].to_vec();
        if !vector_indices.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SPV1 vector indices must be strictly increasing",
            )
            .into());
        }
        vectors.push(SparseVector {
            indices: vector_indices,
            values: values[start..end].to_vec(),
        });
    }

    Ok(vectors)
}

fn load_nbr1(path: &Path) -> Result<(Vec<Vec<i32>>, usize), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"NBR1" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid NBR1 magic").into());
    }

    let rows = read_u32(&mut reader)? as usize;
    let width = read_u32(&mut reader)? as usize;
    let mut neighbors = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(width);
        for _ in 0..width {
            row.push(read_i32(&mut reader)?);
        }
        neighbors.push(row);
    }
    Ok((neighbors, width))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn recall_at_k(results: &[Vec<(u32, f32)>], neighbors: &[Vec<i32>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (result_row, truth_row) in results.iter().zip(neighbors.iter()) {
        let truth: HashSet<u32> = truth_row
            .iter()
            .take(k)
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        for &(id, _) in result_row.iter().take(k) {
            if truth.contains(&id) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f64 / total.max(1) as f64
}
