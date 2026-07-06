//! Microbenchmarks for PQ ADC kernels.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[allow(unused_imports)]
#[path = "../src/pq_simd.rs"]
mod pq_simd;

fn make_flat_lut(num_codebooks: usize) -> Vec<f32> {
    (0..num_codebooks * 16)
        .map(|i| ((i * 17) % 251) as f32 * 0.01)
        .collect()
}

fn make_flat_lut_256(num_codebooks: usize) -> Vec<f32> {
    (0..num_codebooks * 256)
        .map(|i| ((i * 17) % 4093) as f32 * 0.001)
        .collect()
}

fn make_codes(num_vectors: usize, num_codebooks: usize, codebook_size: usize) -> Vec<u8> {
    (0..num_vectors * num_codebooks)
        .map(|i| (i % codebook_size) as u8)
        .collect()
}

fn nested_from_flat(flat_lut: &[f32], num_codebooks: usize) -> Vec<Vec<f32>> {
    (0..num_codebooks)
        .map(|m| {
            let start = m * 16;
            flat_lut[start..start + 16].to_vec()
        })
        .collect()
}

fn bench_fastscan_lut_shape(c: &mut Criterion) {
    let num_vectors = 1_024;
    let num_codebooks = 8;
    let flat_lut = make_flat_lut(num_codebooks);
    let codes = make_codes(num_vectors, num_codebooks, 16);
    let packed = pq_simd::PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);

    let mut group = c.benchmark_group("pq_fastscan_lut_shape");
    group.throughput(Throughput::Elements(num_vectors as u64));

    group.bench_function("build_nested_each_scan", |b| {
        b.iter(|| {
            let nested = nested_from_flat(black_box(&flat_lut), num_codebooks);
            pq_simd::fastscan_batch(black_box(&packed), black_box(&nested))
        })
    });

    group.bench_function("flat_lut", |b| {
        b.iter(|| pq_simd::fastscan_batch_flat(black_box(&packed), black_box(&flat_lut)))
    });

    group.finish();
}

fn bench_standard_adc_lut_shape(c: &mut Criterion) {
    let num_vectors = 16_384;
    let num_codebooks = 25;
    let codebook_size = 256;
    let flat_lut = make_flat_lut_256(num_codebooks);
    let codes = make_codes(num_vectors, num_codebooks, codebook_size);
    let packed_lut = pq_simd::PackedLUT::from_flat(&flat_lut, num_codebooks, codebook_size);

    let mut group = c.benchmark_group("pq_standard_adc_lut_shape");
    group.throughput(Throughput::Elements(num_vectors as u64));

    group.bench_function("build_packed_lut_each_scan", |b| {
        b.iter(|| {
            let packed_lut = pq_simd::PackedLUT::from_flat(
                black_box(&flat_lut),
                black_box(num_codebooks),
                black_box(codebook_size),
            );
            pq_simd::adc_batch_dispatch(
                black_box(&codes),
                black_box(num_codebooks),
                black_box(&packed_lut),
            )
        })
    });

    group.bench_function("packed_lut_dispatch", |b| {
        b.iter(|| {
            pq_simd::adc_batch_dispatch(
                black_box(&codes),
                black_box(num_codebooks),
                black_box(&packed_lut),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fastscan_lut_shape,
    bench_standard_adc_lut_shape
);
criterion_main!(benches);
