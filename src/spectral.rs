//! Spectral sanity helpers (feature-gated).
//!
//! These utilities are intentionally lightweight and *do not* change any default behavior.
//! They can be used by callers that already have eigenvalue sequences (e.g. from PCA/covariance).
//!
//! # Status: building block, no shipped consumer
//!
//! The only in-tree consumer is `estimate_signal_dimensions` in
//! [`crate::adsampling`], which feeds [`crate::adsampling::ADSamplingState::new_auto`].
//! That auto-construction path is exercised only by an internal test; no
//! example, bench, or integration test enables the `rmt-spectral` feature.
//! If you are evaluating the `rmt` dep cost, treat this module as
//! WATCH-tier: the feature exists, the math is correct, but no shipped
//! search path activates it today.

/// Marchenko–Pastur upper bulk edge \(\lambda_+\) for a sample covariance spectrum.
///
/// `ratio` is \(\gamma = p/n\) (features / samples). `sigma_sq` is the assumed noise variance.
///
/// This uses `rmt::marchenko_pastur_support` and returns only \(\lambda_+\).
#[cfg(feature = "rmt-spectral")]
pub fn mp_lambda_plus(ratio: f64, sigma_sq: f64) -> f64 {
    let (_lo, hi) = rmt::marchenko_pastur_support(ratio, sigma_sq);
    hi
}

/// Count eigenvalues strictly above the MP bulk edge (optionally with a multiplicative margin).
///
/// Input eigenvalues must be real-valued; order does not matter.
#[cfg(feature = "rmt-spectral")]
pub fn count_mp_outliers(eigenvalues: &[f64], ratio: f64, sigma_sq: f64, margin: f64) -> usize {
    let thr = mp_lambda_plus(ratio, sigma_sq) * margin.max(1.0);
    eigenvalues
        .iter()
        .filter(|&&x| x.is_finite() && x > thr)
        .count()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rmt-spectral")]
    use super::*;

    #[test]
    #[cfg(feature = "rmt-spectral")]
    fn mp_outlier_count_smoke() {
        // Pretend these are covariance eigenvalues: one clear outlier above the bulk.
        let evals = vec![0.9, 1.1, 1.0, 3.0];
        let ratio = 0.5; // p/n
        let sigma_sq = 1.0;
        let n = count_mp_outliers(&evals, ratio, sigma_sq, 1.0);
        assert!(n >= 1);
    }
}
