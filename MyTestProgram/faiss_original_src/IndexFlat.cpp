/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>
#include <cstring>

namespace faiss {

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances}; // n 个最大堆！
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res);
    } else {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(// 已知所有查询向量的k个id，求距离
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace { // 嵌套匿名空间

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_inner_product(q, (float*)code, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n); // 把浮点数转为二进制字节存储！
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
        : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for // 并行执行批量查询
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) { // q 小于数据库向量的最小值
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) { // q 大于等于数据库向量的最大值
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) // imed 左边可能也有等于 q 的
                i0 = imed;
            else // imed 右边不可能有离 q 更近的
                i1 = imed;
        }
        // i0+1>=i1, i0要么等于i1，要么等于i1
        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]]; // 左边肯定小于等于 q
            float xright = xb[perm[i1]]; // 右边肯定大于等于 q

            if (q - xleft < xright - q) { // q<<1 < xleft + xright
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) { // 不能再往左遍历了
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) { // 不能再往右遍历了
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1 从i1开始往右边找k个
        while (wp < k) { // wp 指第 wp+1 近的向量
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1; // 不足 k 个
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0 从i0开始往左边找k个
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

} // namespace faiss
