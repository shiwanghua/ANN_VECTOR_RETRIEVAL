/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Clustering.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

ClusteringParameters::ClusteringParameters()
        : niter(25),
          nredo(1),
          verbose(false),
          spherical(false),
          int_centroids(false),
          update_index(false),
          frozen_centroids(false),
          min_points_per_centroid(39),
          max_points_per_centroid(256),
          seed(1234),
          decode_block_size(32768) {}
// 39 corresponds to 10000 / 256 -> to avoid warnings on PQ tests with randu10k

Clustering::Clustering(int d, int k) : d(d), k(k) {} // 没有初始化 centroids ！

Clustering::Clustering(int d, int k, const ClusteringParameters& cp)
        : ClusteringParameters(cp), d(d), k(k) {}

// utils 里也实现了一样的 imbalance_factor
static double imbalance_factor(int n, int k, int64_t* assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++)
        hist[assign[i]]++; // assign 记录每个向量属于哪个类

    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}

void Clustering::post_process_centroids() { // 这里并没有考虑冻住的质心
    if (spherical) { // 归一化
        fvec_renorm_L2(d, k, centroids.data());
    }

    if (int_centroids) { // 取整
        for (size_t i = 0; i < centroids.size(); i++)
            centroids[i] = roundf(centroids[i]);
    }
}

void Clustering::train(
        idx_t nx,
        const float* x_in,
        Index& index,
        const float* weights) {
    train_encoded(
            nx,
            reinterpret_cast<const uint8_t*>(x_in),
            nullptr,
            index,
            weights);
}

namespace {

using idx_t = Clustering::idx_t;

idx_t subsample_training_set(
        const Clustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out, // 注意这里是传二级指针，原一级指针还没分配空间
        float** weights_out) {
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }
    std::vector<int> perm(nx); // nx > k * max_points_per_centroid
    rand_perm(perm.data(), nx, clus.seed); // 赋值0~nx-1，打乱顺序
    nx = clus.k * clus.max_points_per_centroid; // 最大质心数
    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;
    for (idx_t i = 0; i < nx; i++) { // 从 x 的原 nx 个向量里随机采样 k * max_points_per_centroid 个作为训练集
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    } // perm[i]没有越界，nx 一定大于 clus.k * clus.max_points_per_centroid
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

/** compute centroids as (weighted) sum of training points
 *
 * @param x            training vectors, size n * code_size (from codec)
 * @param codec        how to decode the vectors (if NULL then cast to float*)
 * @param weights      per-training vector weight, size n (or NULL)
 * @param assign       nearest centroid for each training vector, size n
 * @param k_frozen     do not update the k_frozen first centroids
 * @param centroids    centroid vectors (output only), size k * d
 * @param hassign      histogram of assignments per centroid (size k),
 *                     should be 0 on input
 *
 */

void compute_centroids( // 计算质心，根据新划分的类别数据重新求均值得到新的质心
        size_t d,
        size_t k, // 质心个数
        size_t n,
        size_t k_frozen, // 前几个质心不能修改
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign, // 记录每个新的类里包含多少个向量，原来的 k_frozen 到 k-1 号类的统计个数存储在 hassign 的 0 到 k-k_frozen-1 下标处
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k); // 旧质心归零了！

    size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i]; // assign 的类号是从0开始算的
            assert(ci >= 0 && ci < k + k_frozen);
            ci -= k_frozen;
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi;
                if (!codec) { // 没有编码，直接强制类型转换
                    xi = reinterpret_cast<const float*>(x + i * line_size);
                } else { // 有编码，则解码
                    float* xif = decode_buffer.data();
                    codec->sa_decode(1, x + i * line_size, xif);
                    xi = xif;
                }
                if (weights) {
                    float w = weights[i];
                    hassign[ci] += w; // hassign 的下标是从 k_frozen 开始算的
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * w; // 加权和
                    }
                } else {
                    hassign[ci] += 1.0; // 统计未冻结的类中每个类各有多少个向量
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j]; // 直接求和
                    }
                }
            }
        }
    }

#pragma omp parallel for // 前面质心求了和，接下来应该除以个数，得到均值
    for (idx_t ci = 0; ci < k; ci++) { // 为什么不是到 k-k_frozen ? 因为前面 k 已经减去了 k_frozen!
        if (hassign[ci] == 0) { // 原来的 k_frozen 到 k-1 号类的统计个数存储在 hassign 的 0 到 k-k_frozen-1 下标处
            continue;
        }
        float norm = 1 / hassign[ci]; // 不是除以 hassign[ci]+1，本身原始那个质心没算到 centroids 里，已提前归零了
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one. Takes the same arguments as compute_centroids.
 *
 * @return           nb of spliting operations (larger is worse)
 */
int split_clusters( // 对于每个空类的质心，找一个非空类的质心赋给它，二者都做一些小反向的扰动
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign, // 原来的 k_frozen 到 k-1 号类的统计个数存储在 hassign 的 0 到 k-k_frozen-1 下标处
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */ // 这个类没向量；hassign[ci] 不需要变成 hassign[k_frozen_+ci]，但 ci 记录的是 k_frozen+ci 号类的大小
            size_t cj;
            for (cj = 0; 1; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(*centroids) * d);

            /* small symmetric pertubation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci]; // 不一定是均匀划分
            nsplit++;
        }
    }

    return nsplit; // 空类的个数
}

}; // namespace

void Clustering::train_encoded(
        idx_t nx,
        const uint8_t* x_in,
        const Index* codec, // nullptr
        Index& index, // 借助某种索引来训练，最终质心数据会插入这个index里，某种子类
        const float* weights) {
    FAISS_THROW_IF_NOT_FMT(
            nx >= k,
            "Number of training points (%" PRId64  // lld
            ") should be at least "
            "as large as number of clusters (%zd)",
            nx,
            k);

    FAISS_THROW_IF_NOT_FMT(
            (!codec || codec->d == d),
            "Codec dimension %d not the same as data dimension %d",
            int(codec->d),
            int(d));

    FAISS_THROW_IF_NOT_FMT(
            index.d == d,
            "Index dimension %d not the same as data dimension %d",
            int(index.d),
            int(d));

    double t0 = getmillisecs();

    if (!codec) {
        // Check for NaNs in input data. Normally it is the user's
        // responsibility, but it may spare us some hard-to-debug
        // reports.
        const float* x = reinterpret_cast<const float*>(x_in);
        for (size_t i = 0; i < nx * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]), "input contains NaN's or Inf's");
        }
    }

    const uint8_t* x = x_in;
    std::unique_ptr<uint8_t[]> del1;
    std::unique_ptr<float[]> del3;
    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;

    if (nx > k * max_points_per_centroid) { // 超过聚类的最大容量
        uint8_t* x_new;
        float* weights_new;
        nx = subsample_training_set(
                *this, nx, x, line_size, weights, &x_new, &weights_new);
        del1.reset(x_new);
        x = x_new; // 原 x_in 指针指向的空间在这里得不到释放
        del3.reset(weights_new);
        weights = weights_new; // 原 weights 指针指向的空间在这里得不到释放
    } else if (nx < k * min_points_per_centroid) { // 少于最少质心数
        fprintf(stderr,
                "WARNING clustering %" PRId64
                " points to %zd centroids: "
                "please provide at least %" PRId64 " training points\n",
                nx,
                k,
                idx_t(k) * min_points_per_centroid);
    }

    if (nx == k) {
        // this is a corner case, just copy training set to clusters
        if (verbose) {
            printf("Number of training points (%" PRId64
                   ") same as number of "
                   "clusters, just copying\n",
                   nx);
        }
        centroids.resize(d * k);
        if (!codec) {
            memcpy(centroids.data(), x_in, sizeof(float) * d * k);
        } else {
            codec->sa_decode(nx, x_in, centroids.data());
        }

        // one fake iteration...
        ClusteringIterationStats stats = {0.0, 0.0, 0.0, 1.0, 0};
        iteration_stats.push_back(stats);

        index.reset();
        index.add(k, centroids.data());
        return;
    }

    if (verbose) {
        printf("Clustering %" PRId64
               " points in %zdD to %zd clusters, "
               "redo %d times, %d iterations\n",
               nx,
               d,
               k,
               nredo,
               niter);
        if (codec) {
            printf("Input data encoded in %zd bytes per vector\n",
                   codec->sa_code_size());
        }
    }

    std::unique_ptr<idx_t[]> assign(new idx_t[nx]); // 存储每次迭代的结果
    std::unique_ptr<float[]> dis(new float[nx]);

    // remember best iteration for redo
    bool lower_is_better = index.metric_type != METRIC_INNER_PRODUCT;
    float best_obj = lower_is_better ? HUGE_VALF : -HUGE_VALF; // 0x7f800000
    std::vector<ClusteringIterationStats> best_iteration_stats;
    std::vector<float> best_centroids;

    // support input centroids

    FAISS_THROW_IF_NOT_MSG(
            centroids.size() % d == 0,
            "size of provided input centroids not a multiple of dimension");

    size_t n_input_centroids = centroids.size() / d;

    if (verbose && n_input_centroids > 0) { // 如果 centroids 非空，则为固定质心！
        printf("  Using %zd centroids provided as input (%sfrozen)\n",
               n_input_centroids,
               frozen_centroids ? "" : "not ");
    }

    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n", (getmillisecs() - t0) / 1000.);
    }
    t0 = getmillisecs();

    // temporary buffer to decode vectors during the optimization
    std::vector<float> decode_buffer(codec ? d * decode_block_size : 0); // 一次性解码多少个向量

    for (int redo = 0; redo < nredo; redo++) { // 做 nredo 次随机训练
        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }

        // initialize (remaining) centroids with random points from the dataset
        centroids.resize(d * k); // 不能改变前 d*n_input_centroids(=k) 个数的值！是ProgressiveDimClustering里之前算好的质心
        std::vector<int> perm(nx);

        rand_perm(perm.data(), nx, seed + 1 + redo * 15486557L);

        // 对于 ProgressiveDimClustering，不用随机选，每次直接按上一次的质心做迭代，n_input_centroids==k（第2、3、...次调用时）
        if (!codec) { // 随机选择 k-n_input_centroids 个向量作为非冻住质心
            for (int i = n_input_centroids; i < k; i++) {
                memcpy(&centroids[i * d], x + perm[i] * line_size, line_size);
            }
        } else {
            for (int i = n_input_centroids; i < k; i++) {
                codec->sa_decode(1, x + perm[i] * line_size, &centroids[i * d]);
            }
        }

        post_process_centroids();

        // prepare the index

        if (index.ntotal != 0) {
            index.reset();
        }

        if (!index.is_trained) {
            index.train(k, centroids.data()); // 用这批随机质心训练
        }

        index.add(k, centroids.data());

        // k-means iterations

        float obj = 0;
        for (int i = 0; i < niter; i++) { // 可能已经收敛了但还在迭代...
            double t0s = getmillisecs();

            if (!codec) {
                index.search( // 把训练集当查询向量，找最近的质心
                        nx,
                        reinterpret_cast<const float*>(x),
                        1,
                        dis.get(),
                        assign.get());
            } else {
                // search by blocks of decode_block_size vectors
                size_t code_size = codec->sa_code_size();
                for (size_t i0 = 0; i0 < nx; i0 += decode_block_size) {
                    size_t i1 = i0 + decode_block_size;
                    if (i1 > nx) {
                        i1 = nx;
                    }
                    codec->sa_decode(
                            i1 - i0, x + code_size * i0, decode_buffer.data());
                    index.search(
                            i1 - i0,
                            decode_buffer.data(),
                            1,
                            dis.get() + i0,
                            assign.get() + i0);
                }
            }

            InterruptCallback::check();
            t_search_tot += getmillisecs() - t0s; // 这次迭代的搜索时间

            // accumulate objective
            obj = 0;
            for (int j = 0; j < nx; j++) { // 离质心的总距离
                obj += dis[j];
            }

            // update the centroids
            std::vector<float> hassign(k);

            size_t k_frozen = frozen_centroids ? n_input_centroids : 0;
            compute_centroids( // 根据上面搜索算出来的 assign 计算新的质心
                    d,
                    k,
                    nx,
                    k_frozen, // frozen_centroids 为 true 时，ProgressiveDimClustering 调用该函数会导致 k_frozen==k（第一次调用不会）
                    x,
                    codec,
                    assign.get(),
                    weights,
                    hassign.data(), // 统计非冻住的新类有多少个向量，即 k_frozen 到 k-1 号非冻住类的统计个数存储在 hassign 的 0 到 k-k_frozen-1 下标处
                    centroids.data());

            int nsplit = split_clusters(
                    d, k, nx, k_frozen, hassign.data(), centroids.data());

            // collect statistics
            ClusteringIterationStats stats = {
                    obj, // 总距离，表示本次迭代的聚类效果
                    (getmillisecs() - t0) / 1000.0, // 从开始redo时起经过的秒数，即总训练时间，递增
                    t_search_tot / 1000, // 本次迭代的搜索时间
                    imbalance_factor(nx, k, assign.get()), // 其实 hassign 里已经统计了 k-k_frozen 个数了，这里要重新统计一遍
                    nsplit}; // 空类个数
            iteration_stats.push_back(stats);

            if (verbose) {
                printf("  Iteration %d (%.2f s, search %.2f s): "
                       "objective=%g imbalance=%.3f nsplit=%d       \r",
                       i,
                       stats.time,
                       stats.time_search,
                       stats.obj,
                       stats.imbalance_factor,
                       nsplit);
                fflush(stdout);
            }

            post_process_centroids();

            // add centroids to index for the next iteration (or for output)

            index.reset();
            if (update_index) {
                index.train(k, centroids.data());
            }

            index.add(k, centroids.data());
            InterruptCallback::check();
        } // 迭代结束

        if (verbose)
            printf("\n");
        if (nredo > 1) { // obj 记录的是最后一次迭代后的总距离，每次迭代一定会越来越小吗？每个类求平均后新的质心一定有个更小的总距离？重心规律？
            if ((lower_is_better && obj < best_obj) ||
                (!lower_is_better && obj > best_obj)) {
                if (verbose) { // 这次随机聚类效果比上次好
                    printf("Objective improved: keep new clusters\n");
                }
                best_centroids = centroids;
                best_iteration_stats = iteration_stats; // iteration_stats 不会清零，记录了前面每轮训练的所有迭代结果
                best_obj = obj;
            }
            index.reset(); // nredo>1 时，最后index会被重置
        }
    } // 训练结束
    if (nredo > 1) {
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
        index.reset(); // 这一步多余，前面一定reset了
        index.add(k, best_centroids.data()); // nredo=1时 index 也保存了最后一次迭代后的质心向量
    }
}

Clustering1D::Clustering1D(int k) : Clustering(1, k) {}

Clustering1D::Clustering1D(int k, const ClusteringParameters& cp)
        : Clustering(1, k, cp) {}

void Clustering1D::train_exact(idx_t n, const float* x) {
    const float* xt = x;

    std::unique_ptr<uint8_t[]> del;
    if (n > k * max_points_per_centroid) {
        uint8_t* x_new;
        float* weights_new;
        n = subsample_training_set(
                *this,
                n,
                (uint8_t*)x,
                sizeof(float) * d,
                nullptr,
                &x_new,
                &weights_new);
        del.reset(x_new);
        xt = (float*)x_new;
    }

    centroids.resize(k);
    double uf = kmeans1d(xt, n, k, centroids.data());

    ClusteringIterationStats stats = {0.0, 0.0, 0.0, uf, 0};
    iteration_stats.push_back(stats);
}

float kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids) {
    Clustering clus(d, k);
    clus.verbose = d * n * k > (1L << 30); // 大于10亿个数就打印
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index(d);
    clus.train(n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

/******************************************************************************
 * ProgressiveDimClustering implementation
 ******************************************************************************/

ProgressiveDimClusteringParameters::ProgressiveDimClusteringParameters() {
    progressive_dim_steps = 10;
    apply_pca = true; // seems a good idea to do this by default
    niter = 10;       // reduce nb of iterations per step
}

Index* ProgressiveDimIndexFactory::operator()(int dim) {
    return new IndexFlatL2(dim);
}

ProgressiveDimClustering::ProgressiveDimClustering(int d, int k) : d(d), k(k) {}

ProgressiveDimClustering::ProgressiveDimClustering(
        int d,
        int k,
        const ProgressiveDimClusteringParameters& cp)
        : ProgressiveDimClusteringParameters(cp), d(d), k(k) {}

namespace {

using idx_t = Index::idx_t;

void copy_columns(idx_t n, idx_t d1, const float* src, idx_t d2, float* dest) {
    idx_t d = std::min(d1, d2);
    for (idx_t i = 0; i < n; i++) {
        memcpy(dest, src, sizeof(float) * d);
        src += d1;
        dest += d2;
    }
}

}; // namespace

void ProgressiveDimClustering::train(
        idx_t n,
        const float* x,
        ProgressiveDimIndexFactory& factory) {
    int d_prev = 0;

    PCAMatrix pca(d, d); // 不压缩维度，只提取主成分（维度排序）

    std::vector<float> xbuf;
    if (apply_pca) {
        if (verbose) {
            printf("Training PCA transform\n");
        }
        pca.train(n, x);
        if (verbose) {
            printf("Apply PCA\n");
        }
        xbuf.resize(n * d);
        pca.apply_noalloc(n, x, xbuf.data()); // LinearTransform Ax+b
        x = xbuf.data();
    }

    for (int iter = 0; iter < progressive_dim_steps; iter++) {
        int di = int(pow(d, (1. + iter) / progressive_dim_steps)); // 对前 di 个维度聚类
        if (verbose) {
            printf("Progressive dim step %d: cluster in dimension %d\n",
                   iter,
                   di);
        }
        std::unique_ptr<Index> clustering_index(factory(di));

        Clustering clus(di, k, *this); 
        // *this是ProgressiveDimClustering，
        // 需要转换为ProgressiveDimClusteringParameters，
        // 然后转换为ClusteringParameters

        if (d_prev > 0) {
            // copy warm-start centroids (padded with 0s)
            clus.centroids.resize(k * di); // 这里分配个数太多了吧？389行 n_input_centroids 会等于 k，即所有质心全部被冻住了！除非 frozen_centroids 为 false
            copy_columns( // d_prev<=di，centroids的前 d_prev 列拷贝到 clus.centroids 的前 d_prev 列
                    k, d_prev, centroids.data(), di, clus.centroids.data());
                    // 并不是说把前几轮算出来的质心冻住、不需要变了，每轮都会算k个质心，只是喂的数据维度越来越多而已
        }
        std::vector<float> xsub(n * di); // 考虑每个向量的前 di 个维度
        copy_columns(n, d, x, di, xsub.data()); // 把 x 的前 di 列拷贝到 xsub（紧凑）

        clus.train(n, xsub.data(), *clustering_index.get()); 
        // 普通聚类支持的功能是冻住前 k_frozen=clus.centroids.size()/di 个质心(参考389、486行)
        // 而 ProgressiveDimClustering 需要支持的功能不是冻住所有质心的前 d_prev 个维度
        // 而是基于前 d_prev 个维度算出的k个旧的质心，继续做迭代，但是是基于前di个维度的数据了
        // 所以 frozen_centroids 不是用来实现 ProgressiveDim 的？而且必须为 false ？

        centroids = clus.centroids;
        iteration_stats.insert(
                iteration_stats.end(), // 把新增的迭代状态插入到尾部
                clus.iteration_stats.begin(),
                clus.iteration_stats.end());

        d_prev = di;
    }

    if (apply_pca) {
        if (verbose) {
            printf("Revert PCA transform on centroids\n");
        }
        std::vector<float> cent_transformed(d * k);
        pca.reverse_transform(k, centroids.data(), cent_transformed.data());
        cent_transformed.swap(centroids);
    }
}

} // namespace faiss
