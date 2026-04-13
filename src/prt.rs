//! Probabilistic Routing Test (PRT) for graph-based ANN search.
//!
//! Estimates angular distance between vectors using random subspace projections
//! in o(d) time, avoiding unnecessary full distance computations during graph
//! traversal. Inspired by the Projection-Augmented Graph (PAG) approach
//! (Lu et al., 2026, arXiv:2603.06660).
//!
//! # How it works
//!
//! 1. Pre-compute k random projection vectors (k << d).
//! 2. Project all database vectors into the k-dimensional subspace.
//! 3. During search, project the query once, then compare projected distances
//!    against a threshold to decide whether to compute the full distance.
//! 4. A Test Feedback Buffer (TFB) tightens the threshold across search rounds
//!    by tracking the ratio of false positives, recycling wasted computation.
//!
//! # Usage
//!
//! ```rust,ignore
//! use vicinity::prt::ProbabilisticRoutingTest;
//!
//! let prt = ProbabilisticRoutingTest::new(dimension, 32, Some(42));
//! prt.project_database(&vectors);
//!
//! // During search, for each candidate:
//! let projected_dist = prt.projected_distance(&query_proj, candidate_id);
//! if projected_dist < threshold {
//!     // Worth computing full distance.
//!     let full_dist = l2_distance(query, candidate_vector);
//! }
//! ```

use rand::prelude::*;

/// Probabilistic Routing Test state.
///
/// Maintains random projection vectors and projected database vectors for
/// cheap distance estimation during graph traversal.
#[derive(Debug)]
pub struct ProbabilisticRoutingTest {
    /// Original vector dimension.
    dimension: usize,
    /// Number of projection dimensions (subspace size).
    num_projections: usize,
    /// Projection matrix: num_projections × dimension, stored row-major.
    /// Each row is a random unit vector.
    projections: Vec<f32>,
    /// Projected database vectors: num_vectors × num_projections, stored row-major.
    projected_db: Vec<f32>,
    /// Number of projected database vectors.
    num_vectors: usize,
}

/// Test Feedback Buffer for adaptive threshold tightening.
///
/// Tracks the running ratio of candidates that pass the PRT filter but fail
/// the full distance check (false positives). Uses this to tighten the
/// threshold across search rounds, reducing wasted full-distance computations.
#[derive(Debug, Clone)]
pub struct TestFeedbackBuffer {
    /// Current threshold multiplier (starts at initial_ratio, decreases).
    pub ratio: f32,
    /// Number of PRT passes (candidates that passed the projection test).
    passes: u32,
    /// Number of false positives (passed PRT but failed full distance).
    false_positives: u32,
    /// Decay rate for threshold tightening.
    decay: f32,
    /// Minimum ratio (floor).
    min_ratio: f32,
}

impl ProbabilisticRoutingTest {
    /// Create a new PRT with `num_projections` random projection dimensions.
    ///
    /// `num_projections` controls the accuracy/speed tradeoff:
    /// - 16-32: fast but coarse estimates
    /// - 64-128: slower but more accurate filtering
    ///
    /// Typical: d/8 to d/4 for high-dimensional embeddings.
    pub fn new(dimension: usize, num_projections: usize, seed: Option<u64>) -> Self {
        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Generate random projection vectors (Gaussian, then normalize rows).
        let mut projections = vec![0.0f32; num_projections * dimension];
        for row in 0..num_projections {
            let row_start = row * dimension;
            let mut norm = 0.0f32;
            for j in 0..dimension {
                let v = standard_normal(&mut *rng);
                projections[row_start + j] = v;
                norm += v * v;
            }
            let norm = norm.sqrt();
            if norm > 1e-10 {
                for j in 0..dimension {
                    projections[row_start + j] /= norm;
                }
            }
        }

        Self {
            dimension,
            num_projections,
            projections,
            projected_db: Vec::new(),
            num_vectors: 0,
        }
    }

    /// Project all database vectors into the subspace.
    ///
    /// `vectors` is a flat array of n × dimension f32 values.
    pub fn project_database(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dimension;
        self.num_vectors = n;
        self.projected_db = vec![0.0f32; n * self.num_projections];

        for vec_idx in 0..n {
            let vec_start = vec_idx * self.dimension;
            let vec = &vectors[vec_start..vec_start + self.dimension];
            let proj_start = vec_idx * self.num_projections;

            for p in 0..self.num_projections {
                let row = &self.projections[p * self.dimension..(p + 1) * self.dimension];
                let dot: f32 = row.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum();
                self.projected_db[proj_start + p] = dot;
            }
        }
    }

    /// Project a query vector into the subspace.
    ///
    /// Returns a Vec of `num_projections` projected coordinates.
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.num_projections];
        for (p, out) in result.iter_mut().enumerate() {
            let row = &self.projections[p * self.dimension..(p + 1) * self.dimension];
            *out = row.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
        }
        result
    }

    /// Compute the L2 distance in the projected subspace between the projected
    /// query and the projected database vector at `vec_id`.
    ///
    /// Cost: O(num_projections), which is << O(dimension).
    #[inline]
    pub fn projected_distance(&self, query_proj: &[f32], vec_id: u32) -> f32 {
        let proj_start = vec_id as usize * self.num_projections;
        let db_proj = &self.projected_db[proj_start..proj_start + self.num_projections];
        query_proj
            .iter()
            .zip(db_proj.iter())
            .map(|(&q, &d)| {
                let diff = q - d;
                diff * diff
            })
            .sum()
    }

    /// Check whether a candidate is worth computing the full distance for.
    ///
    /// Returns `true` if the projected distance is below `threshold * tfb.ratio`.
    /// The TFB ratio adapts over the search to reduce false positives.
    #[inline]
    pub fn should_compute_full_distance(
        &self,
        query_proj: &[f32],
        vec_id: u32,
        threshold: f32,
        tfb: &TestFeedbackBuffer,
    ) -> bool {
        let proj_dist = self.projected_distance(query_proj, vec_id);
        // The projected distance underestimates the true distance (by JL lemma).
        // Scale the threshold by the TFB ratio to control false positive rate.
        proj_dist < threshold * tfb.ratio
    }

    /// Number of projection dimensions.
    pub fn num_projections(&self) -> usize {
        self.num_projections
    }

    /// Number of projected database vectors.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

impl TestFeedbackBuffer {
    /// Create a new TFB with initial ratio.
    ///
    /// `initial_ratio`: starting threshold multiplier (typically 1.0-2.0).
    /// Higher = more permissive (fewer skips, higher recall).
    /// `decay`: how fast to tighten on false positives (typically 0.9-0.99).
    pub fn new(initial_ratio: f32, decay: f32) -> Self {
        Self {
            ratio: initial_ratio,
            passes: 0,
            false_positives: 0,
            decay,
            min_ratio: 0.1,
        }
    }

    /// Record a PRT pass that turned out to be useful (full distance < threshold).
    pub fn record_true_positive(&mut self) {
        self.passes += 1;
    }

    /// Record a PRT pass that was wasted (full distance >= threshold).
    /// Tightens the ratio to reduce future false positives.
    pub fn record_false_positive(&mut self) {
        self.passes += 1;
        self.false_positives += 1;
        // Tighten the threshold.
        self.ratio = (self.ratio * self.decay).max(self.min_ratio);
    }

    /// Reset the buffer for a new query.
    pub fn reset(&mut self, initial_ratio: f32) {
        self.ratio = initial_ratio;
        self.passes = 0;
        self.false_positives = 0;
    }

    /// False positive rate so far.
    pub fn false_positive_rate(&self) -> f32 {
        if self.passes == 0 {
            0.0
        } else {
            self.false_positives as f32 / self.passes as f32
        }
    }
}

/// Box-Muller transform for standard normal samples.
fn standard_normal(rng: &mut dyn RngCore) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_project_and_distance() {
        let dim = 32;
        let n_proj = 8;
        let prt = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));

        // Two identical vectors should have projected distance ~0.
        let v1: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut db = v1.clone();
        db.extend(&v1); // 2 copies

        let mut prt = prt;
        prt.project_database(&db);

        let q_proj = prt.project_query(&v1);
        let d0 = prt.projected_distance(&q_proj, 0);
        let d1 = prt.projected_distance(&q_proj, 1);

        assert!(d0 < 1e-6, "self-distance should be ~0, got {}", d0);
        assert!(
            d1 < 1e-6,
            "identical vector distance should be ~0, got {}",
            d1
        );
    }

    #[test]
    fn test_projected_distance_monotonicity() {
        // Projected distances should be correlated with true distances.
        let dim = 64;
        let n_proj = 16;
        let mut prt = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));

        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let near: Vec<f32> = (0..dim).map(|i| i as f32 + 0.1).collect();
        let far: Vec<f32> = (0..dim).map(|i| -(i as f32) * 10.0).collect();

        let mut db = near.clone();
        db.extend(&far);
        prt.project_database(&db);

        let q_proj = prt.project_query(&query);
        let d_near = prt.projected_distance(&q_proj, 0);
        let d_far = prt.projected_distance(&q_proj, 1);

        assert!(
            d_near < d_far,
            "near vector should have smaller projected distance: near={}, far={}",
            d_near,
            d_far
        );
    }

    #[test]
    fn test_tfb_tightening() {
        let mut tfb = TestFeedbackBuffer::new(1.5, 0.9);
        assert!((tfb.ratio - 1.5).abs() < 1e-6);

        // False positives should tighten the ratio.
        tfb.record_false_positive();
        assert!(tfb.ratio < 1.5);
        assert!((tfb.ratio - 1.35).abs() < 1e-6); // 1.5 * 0.9

        // True positives shouldn't change the ratio.
        let ratio_before = tfb.ratio;
        tfb.record_true_positive();
        assert!((tfb.ratio - ratio_before).abs() < 1e-6);

        // FP rate.
        assert!((tfb.false_positive_rate() - 0.5).abs() < 1e-6); // 1 FP out of 2 passes
    }

    #[test]
    fn test_tfb_floor() {
        let mut tfb = TestFeedbackBuffer::new(1.0, 0.5);

        // Many false positives should not drive ratio below min_ratio.
        for _ in 0..100 {
            tfb.record_false_positive();
        }

        assert!(
            tfb.ratio >= 0.1,
            "ratio should not drop below min_ratio, got {}",
            tfb.ratio
        );
    }

    #[test]
    fn test_should_compute_full_distance() {
        let dim = 32;
        let n_proj = 8;
        let mut prt = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));

        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let near: Vec<f32> = (0..dim).map(|i| i as f32 + 0.01).collect();
        let far: Vec<f32> = (0..dim).map(|i| -(i as f32) * 100.0).collect();

        let mut db = near;
        db.extend(&far);
        prt.project_database(&db);

        let q_proj = prt.project_query(&query);
        let tfb = TestFeedbackBuffer::new(1.0, 0.95);

        // With a reasonable threshold, near should pass, far should not.
        let threshold = 10.0;
        let near_pass = prt.should_compute_full_distance(&q_proj, 0, threshold, &tfb);
        let far_pass = prt.should_compute_full_distance(&q_proj, 1, threshold, &tfb);

        assert!(near_pass, "near vector should pass PRT filter");
        assert!(!far_pass, "far vector should be filtered by PRT");
    }

    #[test]
    fn test_projection_preserves_relative_ordering() {
        // Johnson-Lindenstrauss: random projections approximately preserve distances.
        let dim = 128;
        let n_proj = 32;
        let mut prt = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));

        let mut rng = StdRng::seed_from_u64(42);
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        // Create 20 vectors at varying distances from query.
        let mut db = Vec::new();
        let mut true_dists = Vec::new();
        for i in 0..20 {
            let scale = (i + 1) as f32;
            let v: Vec<f32> = query
                .iter()
                .map(|&q| q + rng.random::<f32>() * scale)
                .collect();
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            true_dists.push(d);
            db.extend(v);
        }

        prt.project_database(&db);
        let q_proj = prt.project_query(&query);

        let proj_dists: Vec<f32> = (0..20)
            .map(|i| prt.projected_distance(&q_proj, i))
            .collect();

        // Check rank correlation: the ordering should be roughly preserved.
        // Count concordant pairs (where both orderings agree).
        let n = 20;
        let mut concordant = 0;
        let mut total = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let true_order = true_dists[i] < true_dists[j];
                let proj_order = proj_dists[i] < proj_dists[j];
                if true_order == proj_order {
                    concordant += 1;
                }
                total += 1;
            }
        }

        let kendall_tau = concordant as f32 / total as f32;
        assert!(
            kendall_tau > 0.6,
            "Rank correlation should be > 0.6 (JL property), got {:.2}",
            kendall_tau
        );

        let _ = proj_dists; // suppress unused warning
    }

    #[test]
    fn test_prt_deterministic() {
        let dim = 16;
        let n_proj = 8;
        let prt1 = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));
        let prt2 = ProbabilisticRoutingTest::new(dim, n_proj, Some(42));

        assert_eq!(
            prt1.projections, prt2.projections,
            "same seed = same projections"
        );
    }

    #[test]
    fn test_tfb_reset() {
        let mut tfb = TestFeedbackBuffer::new(1.0, 0.9);
        tfb.record_false_positive();
        tfb.record_false_positive();
        assert!(tfb.ratio < 1.0);

        tfb.reset(1.5);
        assert!((tfb.ratio - 1.5).abs() < 1e-6);
        assert_eq!(tfb.passes, 0);
        assert_eq!(tfb.false_positives, 0);
    }
}
