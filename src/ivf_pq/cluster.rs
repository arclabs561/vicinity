use crate::pq_simd::PackedCodes4bit;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Storage for cluster IDs (compressed or uncompressed).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum ClusterStorage {
    /// Uncompressed IDs (current implementation).
    Uncompressed(Vec<u32>),

    /// Compressed IDs using ROC.
    #[cfg(feature = "id-compression")]
    Compressed {
        data: Vec<u8>,
        num_ids: usize,
        universe_size: u32,
    },
}

/// Cluster (inverted list) containing vector indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Cluster {
    pub(super) storage: ClusterStorage,
    /// Filter bitmask: set of category IDs present in this cluster.
    ///
    /// Bit i is set if any vector in the cluster has category i.
    pub(super) filter_bitmask: u64,
    /// Prepacked 4-bit FastScan codes in this cluster's ID order.
    ///
    /// Faiss stores FastScan codes in block layout instead of packing at query
    /// time. This cache follows that shape for 16-centroid PQ codebooks.
    #[serde(skip)]
    pub(super) fastscan_codes: Option<PackedCodes4bit>,
    /// PQ codes in this cluster's ID order for standard ADC scan.
    ///
    /// The canonical `quantized_codes` array stays in vector-index order for
    /// persistence and scalar lookup. This cache avoids rebuilding a contiguous
    /// scan batch for every probed cluster on the hot search path.
    #[serde(skip)]
    pub(super) adc_codes: Option<Vec<u8>>,
    /// Cache for decompressed IDs (temporary, cleared after use).
    #[cfg(feature = "id-compression")]
    #[serde(skip)]
    #[allow(dead_code)]
    decompressed_cache: Option<Vec<u32>>,
}

impl Cluster {
    /// Create uncompressed cluster.
    pub(super) fn new(ids: Vec<u32>, filter_bitmask: u64) -> Self {
        Self {
            storage: ClusterStorage::Uncompressed(ids),
            filter_bitmask,
            fastscan_codes: None,
            adc_codes: None,
            #[cfg(feature = "id-compression")]
            decompressed_cache: None,
        }
    }

    /// Create compressed cluster.
    #[cfg(feature = "id-compression")]
    pub(super) fn new_compressed(
        ids: Vec<u32>,
        filter_bitmask: u64,
        _compressor: &crate::compression::DeltaVarintCompressor,
        universe_size: u32,
    ) -> Result<Self, crate::compression::CompressionError> {
        // Sort IDs (required for compression)
        let mut sorted_ids = ids;
        sorted_ids.sort();
        sorted_ids.dedup();

        // Compress (self-describing envelope)
        let compressed = crate::compression::compress_set_enveloped(
            &sorted_ids,
            universe_size,
            crate::compression::ChooseConfig::default(),
        )?;

        Ok(Self {
            storage: ClusterStorage::Compressed {
                data: compressed,
                num_ids: sorted_ids.len(),
                universe_size,
            },
            filter_bitmask,
            fastscan_codes: None,
            adc_codes: None,
            decompressed_cache: None,
        })
    }

    /// Get IDs (decompress if needed).
    #[cfg(feature = "id-compression")]
    #[allow(dead_code)]
    pub(super) fn get_ids(&mut self) -> Result<&[u32], crate::compression::CompressionError> {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => Ok(ids),
            ClusterStorage::Compressed {
                data,
                universe_size,
                ..
            } => {
                // Check cache first
                if let Some(ref cached) = self.decompressed_cache {
                    return Ok(cached);
                }

                // Decompress
                let (_choice, u2, decompressed) =
                    crate::compression::decompress_set_enveloped(data)?;
                if u2 != *universe_size {
                    return Err(crate::compression::CompressionError::DecompressionFailed(
                        "universe mismatch in envelope".to_string(),
                    ));
                }

                // Cache (will be cleared after search)
                self.decompressed_cache = Some(decompressed);
                // Safety: just assigned Some on the line above
                #[allow(clippy::unwrap_used)]
                Ok(self.decompressed_cache.as_ref().unwrap())
            }
        }
    }

    /// Get IDs as a borrowed slice (avoids cloning for the uncompressed case).
    pub(super) fn get_ids_ref(&self) -> Cow<'_, [u32]> {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => Cow::Borrowed(ids),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed {
                data,
                universe_size,
                ..
            } => Cow::Owned(
                crate::compression::decompress_set_enveloped(data)
                    .map(|(_choice, u2, ids)| {
                        if u2 == *universe_size {
                            ids
                        } else {
                            Vec::new()
                        }
                    })
                    .unwrap_or_else(|_| Vec::new()),
            ),
        }
    }

    /// Get number of IDs.
    #[allow(dead_code)]
    pub(super) fn len(&self) -> usize {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => ids.len(),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed { num_ids, .. } => *num_ids,
        }
    }

    /// Clear decompression cache (call after search).
    #[cfg(feature = "id-compression")]
    #[allow(dead_code)]
    pub(super) fn clear_cache(&mut self) {
        self.decompressed_cache = None;
    }

    pub(super) fn set_fastscan_codes(&mut self, codes: Option<PackedCodes4bit>) {
        self.fastscan_codes = codes;
    }

    pub(super) fn set_adc_codes(&mut self, codes: Option<Vec<u8>>) {
        self.adc_codes = codes;
    }

    pub(super) fn owned_bytes(&self) -> usize {
        let storage_bytes = match &self.storage {
            ClusterStorage::Uncompressed(ids) => ids.capacity() * std::mem::size_of::<u32>(),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed { data, .. } => data.capacity(),
        };
        let fastscan_bytes = self
            .fastscan_codes
            .as_ref()
            .map(|codes| codes.data.capacity())
            .unwrap_or(0);
        let adc_bytes = self
            .adc_codes
            .as_ref()
            .map(|codes| codes.capacity())
            .unwrap_or(0);
        #[cfg(feature = "id-compression")]
        let cache_bytes = self
            .decompressed_cache
            .as_ref()
            .map(|ids| ids.capacity() * std::mem::size_of::<u32>())
            .unwrap_or(0);
        #[cfg(not(feature = "id-compression"))]
        let cache_bytes = 0;

        storage_bytes + fastscan_bytes + adc_bytes + cache_bytes
    }
}
