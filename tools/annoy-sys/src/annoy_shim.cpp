// Benchmark-only `extern "C"` shim over Spotify's Annoy (Apache-2.0, v1.17.3).
//
// Annoy is header-only, so this translation unit instantiates the Angular/float
// index (`int32_t` ids, `Kiss32Random`) with the single-threaded build policy,
// which is what we compare against: the vector crate's tree builds are
// single-threaded too.
//
// `_K` (which the C++ library derives from the node layout and uses as the leaf
// threshold) is protected, so it is exposed through a subclass for reporting.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "annoylib.h"
#include "kissrandom.h"

namespace {

using Base = Annoy::AnnoyIndex<int32_t, float, Annoy::Angular, Annoy::Kiss32Random,
                               Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

class ShimIndex : public Base {
 public:
  explicit ShimIndex(int f) : Base(f) {}

  int32_t leaf_k() const { return _K; }
};

ShimIndex* as_index(void* handle) { return static_cast<ShimIndex*>(handle); }

}  // namespace

extern "C" {

void* annoy_new(int dimension) { return new ShimIndex(dimension); }

void annoy_free(void* handle) { delete as_index(handle); }

void annoy_set_seed(void* handle, uint32_t seed) { as_index(handle)->set_seed(seed); }

// Annoy's own leaf threshold (`_K`), derived from the node size; reported so the
// comparison can state the effective leaf size instead of assuming one.
int32_t annoy_leaf_k(void* handle) { return as_index(handle)->leaf_k(); }

int annoy_add_item(void* handle, int32_t item, const float* v) {
  return as_index(handle)->add_item(item, v) ? 1 : 0;
}

int annoy_build(void* handle, int trees) {
  // n_threads = 1: single-threaded build, deterministic rng stream.
  return as_index(handle)->build(trees, 1) ? 1 : 0;
}

// Returns the number of ids written (<= k). `search_k == -1` means Annoy's
// default, `k * trees`.
int annoy_search(void* handle, const float* q, int k, int search_k, int32_t* out_ids,
                 float* out_dists) {
  std::vector<int32_t> ids;
  std::vector<float> dists;
  as_index(handle)->get_nns_by_vector(q, static_cast<size_t>(k), search_k, &ids, &dists);
  const size_t n = ids.size();
  for (size_t i = 0; i < n; ++i) {
    out_ids[i] = ids[i];
    out_dists[i] = dists[i];
  }
  return static_cast<int>(n);
}

}  // extern "C"
