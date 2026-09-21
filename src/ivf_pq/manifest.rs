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
    /// Complete payload inventory for generation snapshots. Legacy
    /// direct-directory snapshots predate this field and remain loadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) generation_components: Option<Vec<String>>,
}

impl IVFPQManifest {
    pub(super) fn expected_generation_components(&self) -> Vec<String> {
        let mut components = vec![
            "centroids.bin",
            "clusters.bin",
            "codes.bin",
            "doc_ids.bin",
            "list_codes.bin",
            "list_offsets.bin",
        ];
        if self.raw_vectors_present {
            components.push("raw_vectors.bin");
        }
        components.into_iter().map(String::from).collect()
    }

    pub(super) fn validate_generation_components(
        &self,
        require_inventory: bool,
    ) -> Result<(), String> {
        let Some(components) = &self.generation_components else {
            return if require_inventory {
                Err("IVF-PQ generation manifest is missing its component inventory".into())
            } else {
                Ok(())
            };
        };
        let expected = self.expected_generation_components();
        if components != &expected {
            return Err(format!(
                "IVF-PQ generation component inventory mismatch: expected {expected:?}, got {components:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PersistedFilterMetadata {
    pub(super) doc_id: u32,
    pub(super) metadata: crate::filtering::DocumentMetadata,
}
