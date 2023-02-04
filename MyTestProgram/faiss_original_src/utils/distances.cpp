/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */
// http://www.netlib.org/clapack/cblas/sgemm.c

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

/***************************************************************************
 * Matrix/vector ops
 ***************************************************************************/

/* Compute the L2 norm of a set of nx vectors */
void fvec_norms_L2( // 求平方和，开根号
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = sqrtf(fvec_norm_L2sqr(x + i * d, d));
    }
}

void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++)
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
}

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d); // 平方和

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++)
                xi[j] *= inv_nr;
        }
    }
}

/***************************************************************************
 * KNN functions
 ***************************************************************************/

namespace {

/* Find the nearest neighbors for nx queries in a set of ny vectors */
template <class ResultHandler>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;

            resi.begin(i);

            for (size_t j = 0; j < ny; j++) {
                float ip = fvec_inner_product(x_i, y_j, d);
                resi.add_result(ip, j);
                y_j += d;
            }
            resi.end();
        }
    }
}

template <class ResultHandler>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler; // ResultHandler 可以是 RangeSearchResultHandler， SingleResultHandler 是它的内部类
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res); // 每个线程处理多个查询，每个查询只被1个线程处理
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                float disij = fvec_L2sqr(x_i, y_j, d);
                resi.add_result(disij, j); // 距离，label
                y_j += d;
            }
            resi.end();
        }
    } // 结束时 SingleResultHandler 调用析构函数，自动调用合并结果的函数
}

/** Find the nearest neighbors for nx queries in a set of ny vectors */
template <class ResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }

            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

// distance correction is an operator that can be applied to transform
// the distances
template <class ResultHandler>
void exhaustive_L2sqr_blas(
        const float* x, // 查询向量
        const float* y, // 数据库向量
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const float* y_norms = nullptr) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;    // 4096
    const size_t bs_y = distance_compute_blas_database_bs; // 1024
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx); // 求平方和

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2); // 确保del2析构时能释放 y_norms2 的空间
        fvec_norms_L2sqr(y_norms2, y, d, ny); // y 也求平方和
        y_norms = y_norms2;
    } // y_norms2 退出作用域，但其指向的空间仍有效

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) { // 4096 个查询向量
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) { // 1024 个数据库向量
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                // nyi 个数据库向量，nxi个查询向量
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d; // long
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(), // 存储点积结果
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) { // 每个查询向量
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);
                // 查询向量 i 到 [j0,j1) 数据库向量的点积起始地址

                for (size_t j = j0; j < j1; j++) { // 每个数据库向量
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;
                    // 平方和 - 2 * 点积
                    // 就是L2距离的平方（差的平方和）

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis; // 查询向量 i 到数据库向量 j 的距离平方
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

#ifdef __AVX2__
// an override for AVX2 if only a single closest point is needed.
template <>
void exhaustive_L2sqr_blas<SingleBestResultHandler<CMax<float, int64_t>>>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                _mm_prefetch(ip_line, _MM_HINT_NTA);
                _mm_prefetch(ip_line + 16, _MM_HINT_NTA);

                // constant
                const __m256 mul_minus2 = _mm256_set1_ps(-2);

                // Track 8 min distances + 8 min indices.
                // All the distances tracked do not take x_norms[i]
                //   into account in order to get rid of extra
                //   _mm256_add_ps(x_norms[i], ...) instructions
                //   is distance computations.
                __m256 min_distances =
                        _mm256_set1_ps(res.dis_tab[i] - x_norms[i]);

                // these indices are local and are relative to j0.
                // so, value 0 means j0.
                __m256i min_indices = _mm256_set1_epi32(0);

                __m256i current_indices =
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i indices_delta = _mm256_set1_epi32(8);

                // current j index
                size_t idx_j = 0;
                size_t count = j1 - j0;

                // process 16 elements per loop
                for (; idx_j < (count / 16) * 16; idx_j += 16, ip_line += 16) {
                    _mm_prefetch(ip_line + 32, _MM_HINT_NTA);
                    _mm_prefetch(ip_line + 48, _MM_HINT_NTA);

                    // load values for norms
                    const __m256 y_norm_0 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 0);
                    const __m256 y_norm_1 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 8);

                    // load values for dot products
                    const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
                    const __m256 ip_1 = _mm256_loadu_ps(ip_line + 8);

                    // compute dis = y_norm[j] - 2 * dot(x_norm[i], y_norm[j]).
                    // x_norm[i] was dropped off because it is a constant for a
                    // given i. We'll deal with it later.
                    __m256 distances_0 =
                            _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
                    __m256 distances_1 =
                            _mm256_fmadd_ps(ip_1, mul_minus2, y_norm_1);

                    // compare the new distances to the min distances
                    // for each of the first group of 8 AVX2 components.
                    const __m256 comparison_0 = _mm256_cmp_ps(
                            min_distances, distances_0, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_0, min_distances, comparison_0);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_0));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);

                    // compare the new distances to the min distances
                    // for each of the second group of 8 AVX2 components.
                    const __m256 comparison_1 = _mm256_cmp_ps(
                            min_distances, distances_1, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_1, min_distances, comparison_1);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_1));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);
                }

                // dump values and find the minimum distance / minimum index
                float min_distances_scalar[8];
                uint32_t min_indices_scalar[8];
                _mm256_storeu_ps(min_distances_scalar, min_distances);
                _mm256_storeu_si256(
                        (__m256i*)(min_indices_scalar), min_indices);

                float current_min_distance = res.dis_tab[i];
                uint32_t current_min_index = res.ids_tab[i];

                // This unusual comparison is needed to maintain the behavior
                // of the original implementation: if two indices are
                // represented with equal distance values, then
                // the index with the min value is returned.
                for (size_t jv = 0; jv < 8; jv++) {
                    // add missing x_norms[i]
                    float distance_candidate =
                            min_distances_scalar[jv] + x_norms[i];

                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (distance_candidate < 0)
                        distance_candidate = 0;

                    int64_t index_candidate = min_indices_scalar[jv] + j0;

                    if (current_min_distance > distance_candidate) {
                        current_min_distance = distance_candidate;
                        current_min_index = index_candidate;
                    } else if (
                            current_min_distance == distance_candidate &&
                            current_min_index > index_candidate) {
                        current_min_index = index_candidate;
                    }
                }

                // process leftovers
                for (; idx_j < count; idx_j++, ip_line++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[idx_j + j0] - 2 * ip;
                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (dis < 0)
                        dis = 0;

                    if (current_min_distance > dis) {
                        current_min_distance = dis;
                        current_min_index = idx_j + j0;
                    }
                }

                //
                res.add_result(i, current_min_distance, current_min_index);
            }
        }
        InterruptCallback::check();
    }
}
#endif

} // anonymous namespace

/*******************************************************
 * KNN driver functions
 *******************************************************/

int distance_compute_blas_threshold = 20;
int distance_compute_blas_query_bs = 4096;
int distance_compute_blas_database_bs = 1024;
int distance_compute_min_k_reservoir = 100;

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* ha) {
    if (ha->k < distance_compute_min_k_reservoir) {
        HeapResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k); // ha->nh = nx
        if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res);
        }
    } else {
        ReservoirResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res);
        }
    }
}

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes) {
    float_minheap_array_t heaps = {nx, k, indexes, distances};
    knn_inner_product(x, y, d, nx, ny, &heaps);
}

void knn_L2sqr(
        const float* x, // 查询向量 xq
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t*
                ha, // res = {size_t(n), size_t(k), labels, distances};
        const float* y_norm2) {
    if (ha->k == 1) {
        SingleBestResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids);

        if (nx < distance_compute_blas_threshold) { // 查询个数小于20
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res);
        } else { // 大于等于 20 用 blas
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2);
        }
    } else if (ha->k < distance_compute_min_k_reservoir) { // k < 100
        HeapResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);

        if (nx < distance_compute_blas_threshold) {
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2);
        }
    } else { // k >=100
        // 蓄水池结果处理器
        ReservoirResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2);
        }
    }
}

void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const float* y_norm2) {
    float_maxheap_array_t heaps = {nx, k, indexes, distances};
    knn_L2sqr(x, y, d, nx, ny, &heaps, y_norm2);
}

/***************************************************************************
 * Range search
 ***************************************************************************/

void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res) {
    RangeSearchResultHandler<CMax<float, int64_t>> resh(res, radius); // 析构时调用merge
    if (nx < distance_compute_blas_threshold) { // 小于20个查询时不用blas
        exhaustive_L2sqr_seq(x, y, d, nx, ny, resh);
    } else {
        exhaustive_L2sqr_blas(x, y, d, nx, ny, resh);
    }
}

void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res) {
    RangeSearchResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_inner_product_seq(x, y, d, nx, ny, resh);
    } else {
        exhaustive_inner_product_blas(x, y, d, nx, ny, resh);
    }
}

/***************************************************************************
 * compute a subset of  distances
 ***************************************************************************/

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_inner_products_by_idx(
        float* __restrict ip, // distance
        const float* x, // xq 
        const float* y, // xb
        const int64_t* __restrict ids, /* for y vecs */ // label vector id
        size_t d,
        size_t nx, // nq query vector x
        size_t ny) { // k
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny; // k 个最近的数据库向量的下标构成的数组
        const float* xj = x + j * d; // j-th query vector
        float* __restrict ipj = ip + j * ny; // k distance values
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            ipj[i] = fvec_inner_product(xj, y + d * idsj[i], d);
        }
    }
}

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_L2sqr_by_idx(
        float* __restrict dis,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict disj = dis + j * ny;
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
        }
    }
}

void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_L2sqr(x + d * ix[j], y + d * iy[j], d);
        }
    }
}

void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_inner_product(x + d * ix[j], y + d * iy[j], d);
        }
    }
}

/* Find the nearest neighbors for nx queries in a set of ny vectors
   indexed by ids. May be useful for re-ranking a pre-selected vector list */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res) {
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* idsi = ids + i * ny;
        size_t j;
        float* __restrict simi = res->get_val(i);
        int64_t* __restrict idxi = res->get_ids(i);
        minheap_heapify(k, simi, idxi);

        for (j = 0; j < ny; j++) {
            if (idsi[j] < 0)
                break;
            float ip = fvec_inner_product(x_, y + d * idsi[j], d);

            if (ip > simi[0]) {
                minheap_replace_top(k, simi, idxi, ip, idsi[j]);
            }
        }
        minheap_reorder(k, simi, idxi);
    }
}

void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* __restrict ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res) {
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* __restrict idsi = ids + i * ny;
        float* __restrict simi = res->get_val(i);
        int64_t* __restrict idxi = res->get_ids(i);
        maxheap_heapify(res->k, simi, idxi);
        for (size_t j = 0; j < ny; j++) {
            float disij = fvec_L2sqr(x_, y + d * idsi[j], d);

            if (disij < simi[0]) {
                maxheap_replace_top(k, simi, idxi, disij, idsi[j]);
            }
        }
        maxheap_reorder(res->k, simi, idxi);
    }
}

void pairwise_L2sqr( // 计算 nq 个 d 维查询向量 xq 到 nb 个 d 维数据库向量 xb 的距离，按 xq 优先存在 dis 上
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,   // 相邻两个 xq 之间的距离，如果是紧密存储，ldq=d
        int64_t ldb,   // 相邻两个 xb 之间的距离，如果是紧密存储，ldb=d
        int64_t ldd) { // 第 i+1 个 xq 到第 j 个 xb 的距离地址与第 i 个 xq 到第 j 个 xb 的距离地址之间有 ldd-1 个别的距离地址
    if (nq == 0 || nb == 0)
        return;
    if (ldq == -1)
        ldq = d;
    if (ldb == -1)
        ldb = d;
    if (ldd == -1)
        ldd = nb; // 连续存储，第 i 个 xq 到 nb 个 xb 的距离之后紧挨着第 i+1 个 xq 到 nb 个 xb 的距离

    // store in beginning of distance matrix to avoid malloc
    float* b_norms = dis;

#pragma omp parallel for
    for (int64_t i = 0; i < nb; i++) // 计算每个 xb 的平方和存到了 dis 开始的 nb 个位置，所以下面从 1 号查询向量开始的
        b_norms[i] = fvec_norm_L2sqr(xb + i * ldb, d);

#pragma omp parallel for
    for (int64_t i = 1; i < nq; i++) { // 处理第 1 号之后的每个 xq 向量
        float q_norm = fvec_norm_L2sqr(xq + i * ldq, d); // 第 i 个 xq 的平方和
        for (int64_t j = 0; j < nb; j++)            // 第 i 个 xq 向量到 nb 个 xb 向量的距离赋值为二者的平方和的和，a^2+b^2
            dis[i * ldd + j] = q_norm + b_norms[j]; // 之后会用 blas 计算 “-2ab”
    }

    {
        float q_norm = fvec_norm_L2sqr(xq, d); // 第 0 号 xq 单独处理
        for (int64_t j = 0; j < nb; j++) // 到每个数据库向量的距离，在前面已经被 b_norms 记录了每个 xb 的平方和
            dis[j] += q_norm;  // 加上自己的平方和
    } // 没有动态分配额外内存，同时也尽可能地避免了重复计算

    {
        FINTEGER nbi = nb, nqi = nq, di = d, ldqi = ldq, ldbi = ldb, lddi = ldd;
        float one = 1.0, minus_2 = -2.0; // beta, alpha
        // 对每个 xq 和 xb，计算点积 xq.xb 的 -2 倍并将结果加到 dis 表的相应位置
        // http://www.netlib.org/clapack/cblas/sgemm.c
        // xb 即 A，xq 即 B，dis 即 C
        // ABC按列优先存储！
        // C := alpha*op( A )*op( B ) + beta*C
        sgemm_("Transposed", // 
               "Not transposed",
               &nbi, // xb (di,nbi) 的列数和 dis 的行数是 nbi， 需要转置
               &nqi, // xq (di,nqi) 和 dis 的列数是 nqi
               &di,  // xb 行数、xq 的行数
               &minus_2, // -2
               xb, // xb 实际行数有 ldbi，只取了 di 行做计算
               &ldbi,
               xq, // xq 实际行数有 ldqi，只取了 di 行做计算
               &ldqi,
               &one, // 1.0*dis
               dis,  // (nbi,nqi)，实际行数有 lddi，只计算了这一列的 nbi 个距离（即到这一堆的 ksub 个质心的距离，每一列就是一个查询向量的距离表）
               &lddi);
    }
}

void inner_product_to_L2sqr(
        float* __restrict dis, // （nr1,nr2)
        const float* nr1,
        const float* nr2,
        size_t n1,
        size_t n2) {
#pragma omp parallel for
    for (int64_t j = 0; j < n1; j++) {
        float* disj = dis + j * n2;
        for (size_t i = 0; i < n2; i++)
            disj[i] = nr1[j] + nr2[i] - 2 * disj[i]; // 第 j 行第 i 列
    }
}

} // namespace faiss
