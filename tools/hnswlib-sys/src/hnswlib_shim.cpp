// Thin `extern "C"` layer over `hnswlib::HierarchicalNSW<float>`.
//
// This shim exists only for the parity harness in `tools/parity-rows`. It is not
// a general-purpose binding: the surface is exactly what the measurement needs,
// and every entry point reports failure as a negative count instead of letting a
// C++ exception cross the FFI boundary.
//
// Metric convention (see README.md):
//   0 = squared L2 (hnswlib::L2Space)
//   1 = inner product (hnswlib::InnerProductSpace), i.e. `1 - dot(q, v)`.
// hnswlib has no cosine space. On L2-normalised vectors `1 - dot` is the same
// ordering as cosine distance, so the harness normalises once and uses metric 1.
//
// Labels: hnswlib's `labeltype` is `size_t`; ids are passed and returned as
// `uint64_t`, which is lossless for every id the harness uses (base offsets).

#include "hnswlib/hnswlib.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct Handle {
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* index;
};

int fail(const char* message) {
    g_last_error = (message != nullptr) ? message : "unknown hnswlib error";
    return -1;
}

}  // namespace

extern "C" {

/// Create an index with preallocated capacity.
///
/// `max_elements` is required because hnswlib must allocate its level-0 arena up
/// front; growing it (`resizeIndex`) reallocates and copies the whole graph, so
/// the harness always supplies the corpus size. `metric` follows the convention
/// at the top of this file. Returns null on failure.
void* hnswlib_new(
    size_t dim,
    size_t m,
    size_t ef_construction,
    int metric,
    size_t max_elements) {
    if (dim == 0 || max_elements == 0) {
        fail("hnswlib_new: dim and max_elements must be non-zero");
        return nullptr;
    }

    hnswlib::SpaceInterface<float>* space = nullptr;
    switch (metric) {
        case 0:
            space = new (std::nothrow) hnswlib::L2Space(dim);
            break;
        case 1:
            space = new (std::nothrow) hnswlib::InnerProductSpace(dim);
            break;
        default:
            fail("hnswlib_new: metric must be 0 (L2) or 1 (inner product)");
            return nullptr;
    }
    if (space == nullptr) {
        fail("hnswlib_new: space allocation failed");
        return nullptr;
    }

    try {
        auto* index = new hnswlib::HierarchicalNSW<float>(
            space, max_elements, m, ef_construction, /*random_seed=*/100);
        auto* handle = new (std::nothrow) Handle{space, index};
        if (handle == nullptr) {
            delete index;
            delete space;
            fail("hnswlib_new: handle allocation failed");
            return nullptr;
        }
        return handle;
    } catch (...) {
        delete space;
        fail("hnswlib_new: index construction threw");
        return nullptr;
    }
}

void hnswlib_free(void* raw) {
    auto* handle = static_cast<Handle*>(raw);
    if (handle == nullptr) {
        return;
    }
    delete handle->index;
    delete handle->space;
    delete handle;
}

/// Add one vector under `id`. Returns 0 on success, -1 on failure.
int hnswlib_add(void* raw, uint64_t id, const float* vector) {
    auto* handle = static_cast<Handle*>(raw);
    if (handle == nullptr || vector == nullptr) {
        return fail("hnswlib_add: null handle or vector");
    }
    try {
        hnswlib::Status status = handle->index->addPointNoExceptions(
            vector, static_cast<hnswlib::labeltype>(id), /*replace_deleted=*/false);
        if (!status.ok()) {
            return fail(status.message());
        }
        return 0;
    } catch (...) {
        return fail("hnswlib_add: threw");
    }
}

/// Set the search-time beam width (`ef`). Returns 0 on success, -1 on failure.
int hnswlib_set_ef(void* raw, size_t ef) {
    auto* handle = static_cast<Handle*>(raw);
    if (handle == nullptr) {
        return fail("hnswlib_set_ef: null handle");
    }
    try {
        handle->index->setEf(ef);
        return 0;
    } catch (...) {
        return fail("hnswlib_set_ef: threw");
    }
}

/// Search for the `k` nearest neighbours, closest first.
///
/// Writes at most `k` ids and distances and returns the number written, or -1 on
/// failure. `out_dists` receives hnswlib's raw distance (`1 - dot` for metric 1).
int hnswlib_search(
    void* raw,
    const float* query,
    size_t k,
    uint64_t* out_ids,
    float* out_dists) {
    auto* handle = static_cast<Handle*>(raw);
    if (handle == nullptr || query == nullptr || out_ids == nullptr ||
        out_dists == nullptr) {
        return fail("hnswlib_search: null argument");
    }
    if (k == 0) {
        return 0;
    }
    try {
        auto result = handle->index->searchKnnCloserFirstNoExceptions(
            query, k, /*isIdAllowed=*/nullptr);
        if (!result.ok()) {
            return fail(result.status().message());
        }
        const auto& hits = result.value();
        const size_t written = (hits.size() < k) ? hits.size() : k;
        for (size_t i = 0; i < written; ++i) {
            out_dists[i] = hits[i].first;
            out_ids[i] = static_cast<uint64_t>(hits[i].second);
        }
        return static_cast<int>(written);
    } catch (...) {
        return fail("hnswlib_search: threw");
    }
}

/// Number of elements currently in the index, or 0 for a null handle.
size_t hnswlib_size(void* raw) {
    auto* handle = static_cast<Handle*>(raw);
    if (handle == nullptr) {
        return 0;
    }
    try {
        return handle->index->getCurrentElementCount();
    } catch (...) {
        return 0;
    }
}

/// Last error recorded on this thread. Never null, but may be empty.
const char* hnswlib_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
