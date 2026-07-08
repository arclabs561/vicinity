use super::search::{IVFPQParams, Quantizer};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PersistedIVFPQParams {
    pub(super) num_clusters: usize,
    pub(super) nprobe: usize,
    pub(super) num_codebooks: usize,
    pub(super) codebook_size: usize,
    pub(super) use_opq: bool,
    pub(super) seed: u64,
}

impl From<&IVFPQParams> for PersistedIVFPQParams {
    fn from(params: &IVFPQParams) -> Self {
        Self {
            num_clusters: params.num_clusters,
            nprobe: params.nprobe,
            num_codebooks: params.num_codebooks,
            codebook_size: params.codebook_size,
            use_opq: params.use_opq,
            seed: params.seed,
        }
    }
}

impl PersistedIVFPQParams {
    pub(super) fn into_params(self) -> IVFPQParams {
        IVFPQParams {
            num_clusters: self.num_clusters,
            nprobe: self.nprobe,
            num_codebooks: self.num_codebooks,
            codebook_size: self.codebook_size,
            use_opq: self.use_opq,
            seed: self.seed,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: IVFPQParams::default().compression_threshold,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct IVFPQManifest {
    pub(super) version: u32,
    pub(super) dimension: usize,
    pub(super) num_vectors: usize,
    pub(super) num_centroids: usize,
    pub(super) raw_vectors_present: bool,
    pub(super) params: PersistedIVFPQParams,
    pub(super) quantizer: Quantizer,
    #[serde(default)]
    pub(super) filter_field: Option<String>,
    #[serde(default)]
    pub(super) filter_metadata: Vec<PersistedFilterMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PersistedFilterMetadata {
    pub(super) doc_id: u32,
    pub(super) metadata: crate::filtering::DocumentMetadata,
}
