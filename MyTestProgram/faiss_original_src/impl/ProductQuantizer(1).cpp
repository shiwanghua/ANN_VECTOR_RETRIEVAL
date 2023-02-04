/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ProductQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>

#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

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

/* compute an estimator using look-up tables for typical values of M */
template <typename CT, class C> // CT 不是 8 位就是 16 位 uint
void pq_estimators_from_tables_Mmul4( // 穷举计算 ncodes 个数据库向量里离查询向量最近的 k 个
        int M,
        const CT* codes, // 记录了 ncodes 个数据库向量，具体来说，是存储这些向量的每个短向量堆里所属的那个质心的编号(从 0 开始)
        size_t ncodes,
        const float* __restrict dis_table,
        size_t ksub,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    for (size_t j = 0; j < ncodes; j++) { // 处理 ncodes 个 d 维压缩后的 code(码字) // dis_table 只记录了 1 个查询向量 x 到所有质心的距离，这里为什么 ncodes 个向量共用同一个 dis_table? 因为这些向量是编码压缩后的数据库向量而不是查询向量
        float dis = 0; // 到这个码字的距离
        const float* dt = dis_table;

        for (size_t m = 0; m < M; m += 4) { // M 是 4 的倍数
            // 每轮计算到 4 个短向量的距离；用了 for 的代码迭代展开来加速
            float dism = 0;
            dism = dt[*codes++];
            dt += ksub; // ksub 是每一堆的质心数，每+ksub就指向 到下一堆第一个质心的距离
            dism += dt[*codes++]; // 再偏移 *codes 即第 j 个向量到这一堆的最近质心的编号（第 j 个向量在 m 号堆的编码
            dt += ksub;
            dism += dt[*codes++];
            dt += ksub;
            dism += dt[*codes++];
            dt += ksub;
            dis += dism;
        }

        if (C::cmp(heap_dis[0], dis)) { // 比到堆顶向量的距离小，就替换堆顶向量再自上而下build堆
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j); // 最终存储 k 个 code
        }
    }
}

template <typename CT, class C>
void pq_estimators_from_tables_M4( // M=4的情况，只分了4个堆
        const CT* codes,
        size_t ncodes,
        const float* __restrict dis_table,
        size_t ksub,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    for (size_t j = 0; j < ncodes; j++) {
        float dis = 0;
        const float* dt = dis_table;
        dis = dt[*codes++];
        dt += ksub;
        dis += dt[*codes++];
        dt += ksub;
        dis += dt[*codes++];
        dt += ksub;
        dis += dt[*codes++];

        if (C::cmp(heap_dis[0], dis)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

template <typename CT, class C>
static inline void pq_estimators_from_tables(
        const ProductQuantizer& pq,
        const CT* codes,
        size_t ncodes,
        const float* dis_table,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    if (pq.M == 4) {
        pq_estimators_from_tables_M4<CT, C>(
                codes, ncodes, dis_table, pq.ksub, k, heap_dis, heap_ids);
        return;
    }

    if (pq.M % 4 == 0) {
        pq_estimators_from_tables_Mmul4<CT, C>(
                pq.M, codes, ncodes, dis_table, pq.ksub, k, heap_dis, heap_ids);
        return;
    }

    /* Default is relatively slow */
    const size_t M = pq.M;
    const size_t ksub = pq.ksub;
    for (size_t j = 0; j < ncodes; j++) {
        float dis = 0;
        const float* __restrict dt = dis_table; // 同一个查询向量的距离表
        for (int m = 0; m < M; m++) {
            dis += dt[*codes++]; // 查询向量到第 j 个向量的第 m 号堆上短向量的距离，*codes 是第 j 个向量在第 m 号堆上距离最近的质心的编号
            dt += ksub; // 下一堆距离
        }
        if (C::cmp(heap_dis[0], dis)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

template <class C>
static inline void pq_estimators_from_tables_generic( // codes 需要解码
        const ProductQuantizer& pq,
        size_t nbits,
        const uint8_t* codes, 
        size_t ncodes,
        const float* dis_table,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    const size_t M = pq.M;
    const size_t ksub = pq.ksub;
    for (size_t j = 0; j < ncodes; ++j) {
        PQDecoderGeneric decoder(codes + j * pq.code_size, nbits); // 每个 code 刚好占 code_size 字节，说明存储时码字按字节对齐了？（而实际可能不是8位的整数倍，填充最后一个字节？）下面 set_derived_values 函数里解释了这一点
        float dis = 0;
        const float* __restrict dt = dis_table;
        for (size_t m = 0; m < M; m++) {
            uint64_t c = decoder.decode();
            dis += dt[c];
            dt += ksub;
        }

        if (C::cmp(heap_dis[0], dis)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

/*********************************************
 * PQ implementation
 *********************************************/

ProductQuantizer::ProductQuantizer(size_t d, size_t M, size_t nbits)
        : Quantizer(d, 0), M(M), nbits(nbits), assign_index(nullptr) {
    set_derived_values();
}

ProductQuantizer::ProductQuantizer() : ProductQuantizer(0, 1, 0) {}

void ProductQuantizer::set_derived_values() {
    // quite a few derived values
    FAISS_THROW_IF_NOT_MSG(
            d % M == 0,
            "The dimension of the vector (d) should be a multiple of the number of subquantizers (M)");
    dsub = d / M;
    code_size = (nbits * M + 7) / 8; // 补齐最后一个字节！存储编码时按字节对齐
    ksub = 1 << nbits; // 打满，nbits 位最多可表示多少个质心就设置多少个质心
    centroids.resize(d * ksub); // d 分成 M 份，这里和前面的 dt 不一样，不是按堆优先存储（按列优先，即 1 个长质心的 M 个短质心两两相距 ksub*dsub 个数），而是按向量优先存储（按行优先），连续存储 4 个短向量的编码
    verbose = false;
    train_type = Train_default;
}

void ProductQuantizer::set_params(const float* centroids_, int m) {
    memcpy(get_centroids(m, 0),
           centroids_,
           ksub * dsub * sizeof(centroids_[0])); // 给第 m 堆的质心赋值
}

static void init_hypercube(
        int d,
        int nbits,
        int n,
        const float* x,
        float* centroids) {
    std::vector<float> mean(d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean[j] += x[i * d + j]; // 每维平均值

    float maxm = 0;
    for (int j = 0; j < d; j++) {
        mean[j] /= n;
        if (fabs(mean[j]) > maxm)
            maxm = fabs(mean[j]); // 后面也是+/-，这里没必要取绝对值
    }

    for (int i = 0; i < (1 << nbits); i++) { // 遍历每个长质心进行赋值
        float* cent = centroids + i * d;     // 第 i 号长质心的起始位置
        for (int j = 0; j < nbits; j++)      // 给前 nbits 位即第一个短向量堆的短质心赋值：平均值+/-最大值
            cent[j] = mean[j] + (((i >> j) & 1) ? 1 : -1) * maxm; // i 的第 j 位是 1 就加
        for (int j = nbits; j < d; j++)      // 后 M-1 个短质心的值即各维度的平均值
            cent[j] = mean[j];
    }
}

static void init_hypercube_pca(
        int d,
        int nbits,
        int n,
        const float* x,
        float* centroids) {
    PCAMatrix pca(d, nbits);
    pca.train(n, x);

    for (int i = 0; i < (1 << nbits); i++) {
        float* cent = centroids + i * d;
        for (int j = 0; j < d; j++) {
            cent[j] = pca.mean[j]; // 每个长质心都是同样的值
            float f = 1.0;
            for (int k = 0; k < nbits; k++)
                cent[j] += f * sqrt(pca.eigenvalues[k]) *
                        (((i >> k) & 1) ? 1 : -1) * pca.PCAMat[j + k * d];
        }
    }
}

void ProductQuantizer::train(size_t n, const float* x) {
    if (train_type != Train_shared) { // 堆之间不共享质心，每堆单独训练
        train_type_t final_train_type;
        final_train_type = train_type;
        if (train_type == Train_hypercube ||
            train_type == Train_hypercube_pca) {
            if (dsub < nbits) { // 用 nbits 位来表示 dsub 位
                final_train_type = Train_default;
                printf("cannot train hypercube: nbits=%zd > log2(d=%zd)\n",
                       nbits,
                       dsub);
            }
        }

        float* xslice = new float[n * dsub];
        ScopeDeleter<float> del(xslice);
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < n; j++)
                memcpy(xslice + j * dsub,
                       x + j * d + m * dsub,
                       dsub * sizeof(float)); // 把 x 里第 m 堆短向量拷贝到一维数组上

            Clustering clus(dsub, ksub, cp);

            // we have some initialization for the centroids
            if (final_train_type != Train_default) {
                clus.centroids.resize(dsub * ksub);
            }

            switch (final_train_type) {
                case Train_hypercube:
                    init_hypercube(
                            dsub, nbits, n, xslice, clus.centroids.data());
                    break;
                case Train_hypercube_pca:
                    init_hypercube_pca(
                            dsub, nbits, n, xslice, clus.centroids.data());
                    break;
                case Train_hot_start:
                    memcpy(clus.centroids.data(),
                           get_centroids(m, 0),
                           dsub * ksub * sizeof(float));
                    break;
                default:;
            }

            if (verbose) {
                clus.verbose = true;
                printf("Training PQ slice %d/%zd\n", m, M);
            }
            IndexFlatL2 index(dsub);
            clus.train(n, xslice, assign_index ? *assign_index : index);
            set_params(clus.centroids.data(), m);
        }

    } else { // 所有堆共享质心
        Clustering clus(dsub, ksub, cp);

        if (verbose) {
            clus.verbose = true;
            printf("Training all PQ slices at once\n");
        }

        IndexFlatL2 index(dsub);

        clus.train(n * M, x, assign_index ? *assign_index : index); // 一共有 n*M 个 dsub 维短向量
        for (int m = 0; m < M; m++) {
            set_params(clus.centroids.data(), m); // 每一堆都用同一批质心初始化
        }
    }
}

template <class PQEncoder>
void compute_code(const ProductQuantizer& pq, const float* x, uint8_t* code) { // 编码 x 向量
    std::vector<float> distances(pq.ksub);

    // It seems to be meaningless to allocate std::vector<float> distances.
    // But it is done in order to cope the ineffectiveness of the way
    // the compiler generates the code. Basically, doing something like
    //
    //     size_t min_distance = HUGE_VALF;
    //     size_t idxm = 0;
    //     for (size_t i = 0; i < N; i++) {
    //         const float distance = compute_distance(x, y + i * d, d);
    //         if (distance < min_distance) {
    //            min_distance = distance;
    //            idxm = i;
    //         }
    //     }
    //
    // generates significantly more CPU instructions than the baseline
    //
    //     std::vector<float> distances_cached(N);
    //     for (size_t i = 0; i < N; i++) {
    //         distances_cached[i] = compute_distance(x, y + i * d, d);
    //     }
    //     size_t min_distance = HUGE_VALF;
    //     size_t idxm = 0;
    //     for (size_t i = 0; i < N; i++) {
    //         const float distance = distances_cached[i];
    //         if (distance < min_distance) {
    //            min_distance = distance;
    //            idxm = i;
    //         }
    //     }
    //
    // So, the baseline is faster. This is because of the vectorization.
    // I suppose that the branch predictor might affect the performance as well.
    // So, the buffer is allocated, but it might be unused in
    // manually optimized code. Let's hope that the compiler is smart enough to
    // get rid of std::vector allocation in such a case.

    PQEncoder encoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
        const float* xsub = x + m * pq.dsub; // 第 m 号短向量的起点

        uint64_t idxm = fvec_L2sqr_ny_nearest(
                distances.data(),
                xsub,
                pq.get_centroids(m, 0),
                pq.dsub,
                pq.ksub); // 从 ksub 个质心里找到最近的那个质心并返回它的编号（arm上暴力实现）

        encoder.encode(idxm); // 按位存储这个编号
    }
}

void ProductQuantizer::compute_code(const float* x, uint8_t* code) const {
    switch (nbits) {
        case 8:
            faiss::compute_code<PQEncoder8>(*this, x, code);
            break;

        case 16:
            faiss::compute_code<PQEncoder16>(*this, x, code);
            break;

        default:
            faiss::compute_code<PQEncoderGeneric>(*this, x, code);
            break;
    }
}

template <class PQDecoder>
void decode(const ProductQuantizer& pq, const uint8_t* code, float* x) { // 解码从 code (带偏移) 读到的编号，根据这个编号去取对应的短质心，组成最终的长质心 x
    PQDecoder decoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
        uint64_t c = decoder.decode();
        memcpy(x + m * pq.dsub,
               pq.get_centroids(m, c),
               sizeof(float) * pq.dsub); // 解码出第 m 号堆第 c 号质心
    }
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
    switch (nbits) {
        case 8:
            faiss::decode<PQDecoder8>(*this, code, x);
            break;

        case 16:
            faiss::decode<PQDecoder16>(*this, code, x);
            break;

        default:
            faiss::decode<PQDecoderGeneric>(*this, code, x);
            break;
    }
}

void ProductQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    for (size_t i = 0; i < n; i++) {
        this->decode(code + code_size * i, x + d * i); // 每个编码后的向量code之间是按字节对齐的
    }
}

void ProductQuantizer::compute_code_from_distance_table(
        const float* tab, // 记录了 1 个查询向量到 M*ksub 个质心的距离
        uint8_t* code) const {
    PQEncoderGeneric encoder(code, nbits);
    for (size_t m = 0; m < M; m++) { // 依次处理每一堆
        float mindis = 1e20;
        uint64_t idxm = 0;

        /* Find best centroid */
        for (size_t j = 0; j < ksub; j++) { // 找每一堆的最近质心的编号
            float dis = *tab++; // tab(M,ksub) 遍历了整个距离表
            if (dis < mindis) {
                mindis = dis;
                idxm = j;
            }
        }

        encoder.encode(idxm);
    }
}

void ProductQuantizer::compute_codes_with_assign_index(
        const float* x,
        uint8_t* codes,
        size_t n) {
    FAISS_THROW_IF_NOT(assign_index && assign_index->d == dsub);

    for (size_t m = 0; m < M; m++) {
        assign_index->reset();
        assign_index->add(ksub, get_centroids(m, 0));
        size_t bs = 65536;
        float* xslice = new float[bs * dsub];
        ScopeDeleter<float> del(xslice);
        idx_t* assign = new idx_t[bs];
        ScopeDeleter<idx_t> del2(assign);

        for (size_t i0 = 0; i0 < n; i0 += bs) { // 每次对 bs 个短向量编码
            size_t i1 = std::min(i0 + bs, n);

            for (size_t i = i0; i < i1; i++) {
                memcpy(xslice + (i - i0) * dsub,
                       x + i * d + m * dsub,
                       dsub * sizeof(float));
            }

            assign_index->assign(i1 - i0, xslice, assign); // 求 bs 个短向量离哪个质心最近

            if (nbits == 8) {
                uint8_t* c = codes + code_size * i0 + m;
                for (size_t i = i0; i < i1; i++) {
                    *c = assign[i - i0];
                    c += M; // 每个向量编码为 M 个 8 位整数
                }
            } else if (nbits == 16) {
                uint16_t* c = (uint16_t*)(codes + code_size * i0 + m * 2);
                for (size_t i = i0; i < i1; i++) {
                    *c = assign[i - i0];
                    c += M;
                }
            } else {
                for (size_t i = i0; i < i1; ++i) {
                    uint8_t* c = codes + code_size * i + ((m * nbits) / 8);
                    uint8_t offset = (m * nbits) % 8; // 这个字节里已经用了多少位
                    uint64_t ass = assign[i - i0];

                    PQEncoderGeneric encoder(c, nbits, offset);
                    encoder.encode(ass);
                }
            }
        }
    }
}

// block size used in ProductQuantizer::compute_codes
int product_quantizer_compute_codes_bs = 256 * 1024;

void ProductQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    // process by blocks to avoid using too much RAM
    size_t bs = product_quantizer_compute_codes_bs;
    if (n > bs) {
        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);
            compute_codes(x + d * i0, codes + code_size * i0, i1 - i0);
        }
        return;
    }

    if (dsub < 16) { // simple direct computation

#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            compute_code(x + i * d, codes + i * code_size);

    } else { // worthwile to use BLAS
        float* dis_tables = new float[n * ksub * M];
        ScopeDeleter<float> del(dis_tables);
        compute_distance_tables(n, x, dis_tables);

#pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
            uint8_t* code = codes + i * code_size;
            const float* tab = dis_tables + i * ksub * M;
            compute_code_from_distance_table(tab, code);
        }
    }
}

void ProductQuantizer::compute_distance_table(const float* x, float* dis_table)
        const {
    size_t m;

    for (m = 0; m < M; m++) { // x 到 M*ksub个质心的距离
        fvec_L2sqr_ny(
                dis_table + m * ksub,
                x + m * dsub,
                get_centroids(m, 0),
                dsub,
                ksub);
    }
}

void ProductQuantizer::compute_inner_prod_table(
        const float* x,
        float* dis_table) const {
    size_t m;

    for (m = 0; m < M; m++) {
        fvec_inner_products_ny(
                dis_table + m * ksub,
                x + m * dsub,
                get_centroids(m, 0),
                dsub,
                ksub);
    }
}

void ProductQuantizer::compute_distance_tables(
        size_t nx,
        const float* x,
        float* dis_tables) const {
#if defined(__AVX2__) || defined(__aarch64__)
    if (dsub == 2 && nbits < 8) { // interesting for a narrow range of settings
        compute_PQ_dis_tables_dsub2(
                d, ksub, centroids.data(), nx, x, false, dis_tables);
    } else
#endif
            if (dsub < 16) {

#pragma omp parallel for
        for (int64_t i = 0; i < nx; i++) {
            compute_distance_table(x + i * d, dis_tables + i * ksub * M);
        }

    } else { // use BLAS

        for (int m = 0; m < M; m++) {
            pairwise_L2sqr(
                    dsub,
                    nx,
                    x + dsub * m,
                    ksub,
                    centroids.data() + m * dsub * ksub, // 第 m 号堆质心
                    dis_tables + ksub * m, // 第一个距离表的第 m 号堆的起始位置，每个表的大小为 ksub*M
                    d,    // 相邻两个 x 之间实际相差 d 个数
                    dsub, // 相邻两个质心是紧挨着连续存储的
                    ksub * M); // 计算 nx 个向量的第 m 号 dsub 维短向量到第 m 号堆的 ksub 个质心的距离
        }
    }
}

void ProductQuantizer::compute_inner_prod_tables(
        size_t nx,
        const float* x,
        float* dis_tables) const {
#if defined(__AVX2__) || defined(__aarch64__)
    if (dsub == 2 && nbits < 8) {
        compute_PQ_dis_tables_dsub2(
                d, ksub, centroids.data(), nx, x, true, dis_tables);
    } else
#endif
            if (dsub < 16) {

#pragma omp parallel for
        for (int64_t i = 0; i < nx; i++) {
            compute_inner_prod_table(x + i * d, dis_tables + i * ksub * M);
        }

    } else { // use BLAS

        // compute distance tables
        for (int m = 0; m < M; m++) {
            FINTEGER ldc = ksub * M, nxi = nx, ksubi = ksub, dsubi = dsub,
                     di = d;
            float one = 1.0, zero = 0;

            sgemm_("Transposed",
                   "Not transposed",
                   &ksubi, // 每堆质心个数、数据库向量个数
                   &nxi,
                   &dsubi,
                   &one,
                   &centroids[m * dsub * ksub],
                   &dsubi,
                   x + dsub * m,
                   &di,
                   &zero, // 不需要保存dis上原来的值
                   dis_tables + ksub * m,
                   &ldc); // 无需调用 pairwise_L2sqr 算平方和了，直接求点积
        }
    }
}

template <class C>
static void pq_knn_search_with_tables( // 对每个 res 里的查询向量，穷举搜索 ncodes 个数据库向量，记录最近的 k 个...
        const ProductQuantizer& pq,
        size_t nbits,
        const float* dis_tables,
        const uint8_t* codes,
        const size_t ncodes,
        HeapArray<C>* res,
        bool init_finalize_heap) {
    size_t k = res->k, nx = res->nh;
    size_t ksub = pq.ksub, M = pq.M;

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) { // 每个查询向量
        /* query preparation for asymmetric search: compute look-up tables */
        const float* dis_table = dis_tables + i * ksub * M;

        /* Compute distances and keep smallest values */
        int64_t* __restrict heap_ids = res->ids + i * k;
        float* __restrict heap_dis = res->val + i * k;

        if (init_finalize_heap) {
            heap_heapify<C>(k, heap_dis, heap_ids);
        }

        switch (nbits) {
            case 8:
                pq_estimators_from_tables<uint8_t, C>(
                        pq, codes, ncodes, dis_table, k, heap_dis, heap_ids);
                break;

            case 16:
                pq_estimators_from_tables<uint16_t, C>(
                        pq,
                        (uint16_t*)codes,
                        ncodes,
                        dis_table,
                        k,
                        heap_dis,
                        heap_ids);
                break;

            default:
                pq_estimators_from_tables_generic<C>( // 数据库向量需要解码
                        pq,
                        nbits,
                        codes,
                        ncodes,
                        dis_table,
                        k,
                        heap_dis,
                        heap_ids);
                break;
        }

        if (init_finalize_heap) {
            heap_reorder<C>(k, heap_dis, heap_ids); // 按距离从小到大排序
        }
    }
}

void ProductQuantizer::search(
        const float* __restrict x,
        size_t nx,
        const uint8_t* codes,
        const size_t ncodes,
        float_maxheap_array_t* res,
        bool init_finalize_heap) const {
    FAISS_THROW_IF_NOT(nx == res->nh);
    std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
    compute_distance_tables(nx, x, dis_tables.get());

    pq_knn_search_with_tables<CMax<float, int64_t>>(
            *this,
            nbits,
            dis_tables.get(),
            codes,
            ncodes,
            res,
            init_finalize_heap);
}

void ProductQuantizer::search_ip(
        const float* __restrict x,
        size_t nx,
        const uint8_t* codes,
        const size_t ncodes,
        float_minheap_array_t* res,
        bool init_finalize_heap) const {
    FAISS_THROW_IF_NOT(nx == res->nh);
    std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
    compute_inner_prod_tables(nx, x, dis_tables.get());

    pq_knn_search_with_tables<CMin<float, int64_t>>(
            *this,
            nbits,
            dis_tables.get(),
            codes,
            ncodes,
            res,
            init_finalize_heap);
}

static float sqr(float x) {
    return x * x;
}

void ProductQuantizer::compute_sdc_table() {
    sdc_table.resize(M * ksub * ksub); // 每个堆里任意两个质心之间的距离（有一半空间浪费）

    if (dsub < 4) {
#pragma omp parallel for
        for (int mk = 0; mk < M * ksub; mk++) { // 遍历每个质心
            // allow omp to schedule in a more fine-grained way
            // `collapse` is not supported in OpenMP 2.x
            int m = mk / ksub;
            int k = mk % ksub;
            const float* cents = centroids.data() + m * ksub * dsub; // 第 m 号堆的第一个质心
            const float* centi = cents + k * dsub; // 第 mk 个质心
            float* dis_tab = sdc_table.data() + m * ksub * ksub; // 第 m 堆的对称距离表
            fvec_L2sqr_ny(dis_tab + k * ksub, centi, cents, dsub, ksub); // 求第 mk 个质心到第 m 号堆内每个质心的距离
        }
    } else {
        // NOTE: it would disable the omp loop in pairwise_L2sqr
        // but still accelerate especially when M >= 4
#pragma omp parallel for // 维度比较多的时候用 blas 算
        for (int m = 0; m < M; m++) {
            const float* cents = centroids.data() + m * ksub * dsub;
            float* dis_tab = sdc_table.data() + m * ksub * ksub;
            pairwise_L2sqr(
                    dsub, ksub, cents, ksub, cents, dis_tab, dsub, dsub, ksub);
        }
    }
}

void ProductQuantizer::search_sdc(
        const uint8_t* qcodes, // 需预先对查询向量编码，得到每个查询向量在每个堆上距离最近的质心编号
        size_t nq,
        const uint8_t* bcodes,
        const size_t nb,
        float_maxheap_array_t* res,
        bool init_finalize_heap) const {
    FAISS_THROW_IF_NOT(sdc_table.size() == M * ksub * ksub);
    FAISS_THROW_IF_NOT(nbits == 8);
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nq; i++) { // 并行处理每个查询向量 i
        /* Compute distances and keep smallest values */
        idx_t* heap_ids = res->ids + i * k;
        float* heap_dis = res->val + i * k;
        const uint8_t* qcode = qcodes + i * code_size;

        if (init_finalize_heap)
            maxheap_heapify(k, heap_dis, heap_ids);

        const uint8_t* bcode = bcodes; // 记录每个数据库向量在 m 个堆上的编码（最近质心编号）
        for (size_t j = 0; j < nb; j++) { // 遍历每个数据库向量 j
            float dis = 0;
            const float* tab = sdc_table.data();
            for (int m = 0; m < M; m++) { // 计算 ij 在 m 号堆上的对称距离
                dis += tab[bcode[m] + qcode[m] * ksub]; // 核心代码：第 m 号堆第 qcode[m] 号质心到第 bcode[m] 号质心的对称距离
                tab += ksub * ksub;
            }
            if (dis < heap_dis[0]) {
                maxheap_replace_top(k, heap_dis, heap_ids, dis, j);
            }
            bcode += code_size;
        }

        if (init_finalize_heap)
            maxheap_reorder(k, heap_dis, heap_ids);
    }
}

} // namespace faiss
