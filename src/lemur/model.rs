//! Two-layer MLP encoder for LEMUR.
//!
//! Architecture: `psi(x) = LayerNorm(GELU(W'x + b))`
//!
//! W' has shape `[hidden_dim, input_dim]`, b has shape `[hidden_dim]`.
//! The outer linear layer W is only used during training (for OLS targets)
//! and is not needed at inference.

/// Two-layer MLP encoder.
///
/// Loads pre-trained weights and runs the forward pass for query/document
/// token encoding. No training support: train externally (e.g., PyTorch)
/// and load the weights.
pub struct LemurEncoder {
    /// Weight matrix W' of shape `[hidden_dim, input_dim]`, row-major.
    w: Vec<f32>,
    /// Bias vector b of shape `[hidden_dim]`.
    b: Vec<f32>,
    /// LayerNorm gamma (scale) of shape `[hidden_dim]`.
    ln_gamma: Vec<f32>,
    /// LayerNorm beta (shift) of shape `[hidden_dim]`.
    ln_beta: Vec<f32>,
    /// Input dimension.
    input_dim: usize,
    /// Hidden dimension (d').
    hidden_dim: usize,
}

impl LemurEncoder {
    /// Create an encoder from raw weight arrays.
    ///
    /// # Arguments
    /// * `input_dim` - Token embedding dimension
    /// * `hidden_dim` - MLP hidden dimension (d', typically 2048)
    /// * `w` - Weight matrix of shape `[hidden_dim, input_dim]`, row-major
    /// * `b` - Bias vector of shape `[hidden_dim]`
    /// * `ln_gamma` - LayerNorm scale of shape `[hidden_dim]`
    /// * `ln_beta` - LayerNorm shift of shape `[hidden_dim]`
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        w: Vec<f32>,
        b: Vec<f32>,
        ln_gamma: Vec<f32>,
        ln_beta: Vec<f32>,
    ) -> Result<Self, crate::RetrieveError> {
        if w.len() != hidden_dim * input_dim {
            return Err(crate::RetrieveError::InvalidParameter(format!(
                "w must be {} elements (hidden_dim * input_dim), got {}",
                hidden_dim * input_dim,
                w.len()
            )));
        }
        if b.len() != hidden_dim {
            return Err(crate::RetrieveError::InvalidParameter(format!(
                "b must be {} elements, got {}",
                hidden_dim,
                b.len()
            )));
        }
        if ln_gamma.len() != hidden_dim || ln_beta.len() != hidden_dim {
            return Err(crate::RetrieveError::InvalidParameter(
                "ln_gamma and ln_beta must have hidden_dim elements".into(),
            ));
        }
        Ok(Self {
            w,
            b,
            ln_gamma,
            ln_beta,
            input_dim,
            hidden_dim,
        })
    }

    /// Create an encoder with random weights for tests and fixtures.
    ///
    /// Random weights produce random retrieval quality. Use [`LemurEncoder::new`]
    /// with externally trained weights for indexing or search experiments.
    #[cfg(any(test, feature = "lemur-fixtures"))]
    #[doc(hidden)]
    #[deprecated(
        note = "random weights are test fixtures; use LemurEncoder::new with trained weights for retrieval"
    )]
    pub fn random(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / input_dim as f64).sqrt() as f32; // Kaiming init

        let w: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let b = vec![0.0f32; hidden_dim];
        let ln_gamma = vec![1.0f32; hidden_dim];
        let ln_beta = vec![0.0f32; hidden_dim];

        Self {
            w,
            b,
            ln_gamma,
            ln_beta,
            input_dim,
            hidden_dim,
        }
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Forward pass: psi(x) = LayerNorm(GELU(W'x + b)).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.input_dim);

        // Linear: h = W'x + b
        let mut h = self.b.clone();
        for (i, h_val) in h.iter_mut().enumerate().take(self.hidden_dim) {
            let row = &self.w[i * self.input_dim..(i + 1) * self.input_dim];
            let sum: f32 = row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum();
            *h_val += sum;
        }

        // GELU activation
        for v in h.iter_mut() {
            *v = gelu(*v);
        }

        // LayerNorm
        let mean: f32 = h.iter().sum::<f32>() / self.hidden_dim as f32;
        let var: f32 =
            h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / self.hidden_dim as f32;
        let std = (var + 1e-5).sqrt();

        for ((h_val, &gamma), &beta) in h
            .iter_mut()
            .zip(self.ln_gamma.iter())
            .zip(self.ln_beta.iter())
        {
            *h_val = (*h_val - mean) / std * gamma + beta;
        }

        h
    }
}

/// GELU activation: x * Phi(x) where Phi is the standard normal CDF.
/// Uses the fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, deprecated)]
mod tests {
    use super::*;

    #[test]
    fn forward_output_dimension() {
        let enc = LemurEncoder::random(128, 256, 42);
        let x = vec![0.1f32; 128];
        let out = enc.forward(&x);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn forward_deterministic() {
        let enc = LemurEncoder::random(64, 128, 99);
        let x = vec![0.5f32; 64];
        let a = enc.forward(&x);
        let b = enc.forward(&x);
        assert_eq!(a, b);
    }

    #[test]
    fn gelu_zero_is_zero() {
        assert!((gelu(0.0)).abs() < 1e-6);
    }

    #[test]
    fn gelu_positive_is_positive() {
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(5.0) > 4.9); // approaches identity for large x
    }
}
