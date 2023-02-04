/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVF.h>

#include <omp.h>
#include <mutex>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <memory>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

/*****************************************
 * Level1Quantizer implementation
 ******************************************/

Level1Quantizer::Level1Quantizer(Index* quantizer, size_t nlist)
        : quantizer(quantizer),
          nlist(nlist),
          quantizer_trains_alone(0),
          own_fields(false),
          clustering_index(nullptr) {
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer()
        : quantizer(nullptr),
          nlist(0),
          quantizer_trains_alone(0),
          own_fields(false),
          clustering_index(nullptr) {}

Level1Quantizer::~Level1Quantizer() {
    if (own_fields)
        delete quantizer;
}

void Level1Quantizer::train_q1(
        size_t n,
        const float* x,
        bool verbose,
        MetricType metric_type) {
    size_t d = quantizer->d;
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf("IVF quantizer trains alone...\n");
        quantizer->train(n, x);
        quantizer->verbose = verbose;
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf("Training level-1 quantizer on %zd vectors in %zdD\n", n, d);

        Clustering clus(d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train(n, x, *clustering_index);
            quantizer->add(nlist, clus.centroids.data());
        } else {
            clus.train(n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) { // 聚类训练+量化器本身训练+插入质心
        if (verbose) {
            printf("Training L2 quantizer on %zd vectors in %zdD%s\n",
                   n,
                   d,
                   clustering_index ? "(user provided index)" : "");
        }
        // also accept spherical centroids because in that case
        // L2 and IP are equivalent
        FAISS_THROW_IF_NOT(
                metric_type == METRIC_L2 ||
                (metric_type == METRIC_INNER_PRODUCT && cp.spherical));

        Clustering clus(d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner(d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose) {
            printf("Adding centroids to quantizer\n");
        }
        if (!quantizer->is_trained) {
            if (verbose) {
                printf("But training it first on centroids table...\n");
            }
            quantizer->train(nlist, clus.centroids.data());
        }
        quantizer->add(nlist, clus.centroids.data());
    }
}

size_t Level1Quantizer::coarse_code_size() const { // 一个类的编号需要多少个字节表示
    size_t nl = nlist - 1; // nlist<=256时只需要一个字节
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

// 对倒排表号编码存到code上，因为按字节存储，避免了没有正好3、5个字节的数据基本类型这样的问题，不过3个字节意味着16777216个质心了
void Level1Quantizer::encode_listno(Index::idx_t list_no, uint8_t* code) const {
    // little endian
    size_t nl = nlist - 1;
    while (nl > 0) { // list_no肯定小于等于nl，其实list_no等于0时就可以停止了（如果分配空间时进行了初始化的话）
        *code++ = list_no & 0xff; // 小端，先存储低8位！
        list_no >>= 8;
        nl >>= 8;
    }
}

Index::idx_t Level1Quantizer::decode_listno(const uint8_t* code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8; // 可以记录 coarse_code_size 的返回值，不需要每次都循环计算 nl
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}

/*****************************************
 * IndexIVF implementation
 ******************************************/

IndexIVF::IndexIVF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        MetricType metric)
        : Index(d, metric),
          Level1Quantizer(quantizer, nlist),
          invlists(new ArrayInvertedLists(nlist, code_size)),
          own_invlists(true),
          code_size(code_size),
          nprobe(1),
          max_codes(0),
          parallel_mode(0) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
}

IndexIVF::IndexIVF()
        : invlists(nullptr),
          own_invlists(false),
          code_size(0),
          nprobe(1),
          max_codes(0),
          parallel_mode(0) {}

void IndexIVF::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get()); // 计算每个向量离哪个质心最近
    add_core(n, x, xids, coarse_idx.get());
}

// 新增向量所属倒排列表号 和 新增向量被压缩后的码字 按序放在同一个数组codes上了
// include_listno 为 true 时才会这样存储，但它默认是 false，但 sa_code_size 一定返回倒排列表号的 size 与压缩码字的 size 之和
void IndexIVF::add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids) { // add 压缩后的向量
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs); // 把映射关系加进去
    }
    ntotal += n;
}

// 把向量压缩编码后并行插入到各自所属的倒排列表
void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx) {
    // do some blocking to avoid excessive allocs
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("   IndexIVF::add_with_ids %" PRId64 ":%" PRId64 "\n",
                       i0,
                       i1);
            }
            add_core(
                    i1 - i0,
                    x + i0 * d,
                    xids ? xids + i0 : nullptr,
                    coarse_idx + i0);
        }
        return;
    }
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0) // 什么情况会搜出-1？
            nminus1++;
    }

    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]); // 没有存储倒排表的序号，但 encode_vectors 中可能存储，需要小心
    encode_vectors(n, x, coarse_idx, flat_codes.get()); // 纯虚函数，在子类比如 IVFPQ、IndexIVFFlat 中实现（进行编码）

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) { // 轮训分配给线程
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no, id, flat_codes.get() + i * code_size); // 只有压缩后的编码

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
               nadd,
               n,
               nminus1);
    }

    ntotal += n; // 包括有效和无效占位的编码向量
}

void IndexIVF::make_direct_map(bool b) {
    if (b) {
        direct_map.set_type(DirectMap::Array, invlists, ntotal);
    } else {
        direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    }
}

void IndexIVF::set_direct_map_type(DirectMap::Type type) {
    direct_map.set_type(type, invlists, ntotal);
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    const size_t nprobe = std::min(nlist, this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries // 查询最近的 nprobe 个质心
    auto sub_search_func = [this, k, nprobe](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get()); // 质心存在索引 quantizer 里
        double t1 = getmillisecs();

        invlists->prefetch_lists(idx.get(), n * nprobe); // 预取 n*nprobe 个倒排列表，没看懂怎么取的

        search_preassigned( // 二级搜索
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                nullptr,
                ivf_stats);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0; // 一级搜索时间
        ivf_stats->search_time += t2 - t0; // 二级搜索时间，这里应该是 t2-t1，和 range_search 不一致
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) { // (0)&(111...101111111111)
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats; // 没用到
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt; // 处理 [i0,i1) 号查询向量
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle paralellization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_preassigned( // 二级搜索
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params, // 可选的可修改的两个搜索条件/搜索参数
        IndexIVFStats* ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t max_codes = params ? params->max_codes : this->max_codes;

    size_t nlistv = 0, ndis = 0, nheap = 0; // ndis 记录计算了多少次距离、比较了多少个向量

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT; // ~PARALLEL_MODE_NO_HEAP_INIT=-1025, 和 0123 进行与运算后不还是 0123 吗？
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT); // parallel_mode=0时初始化堆

    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1); 
    // parallel_mode=0时不并行；
    // parallel_mode=1时对查询内部每个倒排表的搜索进行并行；
    // parallel_mode=2时对查询向量并行，每个查询内部也并行搜索聚类
    // parallel_mode=3时只并行查询向量

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        InvertedListScanner* scanner = get_InvertedListScanner(store_pairs);
        ScopeDeleter1<InvertedListScanner> del(scanner);

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // intialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) { // 插入 k 个向量到空堆里
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi); // k0=0，就是进行了一个初始化赋值，即默认答案
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };
        // 往已经有k个元素的堆里插入k个元素，并行搜索倒排列表后合并线程部分结果时调用
        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) { // 探索完倒排列表后，把堆数组按距离从小到大排序（每次把堆顶元素放到最后一个位置）
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi // 返回这个倒排列表的长度
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            size_t list_size = invlists->list_size(key);

            // don't waste time on empty lists
            if (list_size == 0) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i); // 主要是保存一下自己负责的倒排列表号

            nlistv++;

            try {
                InvertedLists::ScopedCodes scodes(invlists, key); // 为什么要删除invlists里的倒排列表？

                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t* ids = nullptr;

                if (!store_pairs) {
                    sids.reset(new InvertedLists::ScopedIds(invlists, key));
                    ids = sids->get();
                }

                nheap += scanner->scan_codes(
                        list_size, scodes.get(), ids, simi, idxi, k); // 返回入堆了多少次

            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }

            return list_size;
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) { // 前面 do_parallel 区分出来了 1、3 模式
#pragma omp for // 如果 do_parallel 为 false，这里应该不会并行
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k; // 每个线程修改的是不同位置，不需要同步
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi); // simi总长度很小，可放在一个 L1 cache 里，并行修改时不能共享，并行可能没效果

                idx_t nscan = 0; // 一共探索了多少个向量

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik], // 第 i 个查询向量的第 ik+1 近的倒排列表的表号
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi);

                    if (max_codes && nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi); // 按距离增序排序

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k); // do_parallel 为 true 启动了并行，因此会生成多个local数组！
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) { // 这里没启用并行？！
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data()); // 修改的是不同数组，也不会有并行错误

                    // can't do the test on max_codes // max_codes 同步开销很大
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier  // 为什么要等待每个线程都算完？为了保证初始化函数完成了
#pragma omp critical // 把每个线程的部分结果都加到最终答案里，需要异步插入
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) { // 查询向量、倒排列表双并行
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) { // 初始化 n 个堆
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe; // 第 i 个查询向量的第 j+1 近的倒排列表
                size_t j = ij % nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data());
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
    }
}

void IndexIVF::range_search(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result) const {
    const size_t nprobe = std::min(nlist, this->nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(nx, x, nprobe, coarse_dis.get(), keys.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * nprobe);

    range_search_preassigned(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            nullptr,
            &indexIVF_stats);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : this->max_codes;

    size_t nlistv = 0, ndis = 0; // 没有 nheap 了，小于等于 radius 的都是对的

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * nx > 1);
    // parallel_mode=0时并行查询向量；
    // parallel_mode=1时对单个查询向量内部倒排表的搜索进行并行；
    // parallel_mode=2时对查询向量并行，每个查询内部也并行搜索聚类
    // parallel_mode不能等于3

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result); // 每个线程都会声明
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs)); // IVFFlat、IVFPQ 重载
        FAISS_THROW_IF_NOT(scanner.get());
        all_pres[omp_get_thread_num()] = &pres; // 用于最后合并

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = keys[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);
            const size_t list_size = invlists->list_size(key);

            if (list_size == 0)
                return;

            try {
                InvertedLists::ScopedCodes scodes(invlists, key);
                InvertedLists::ScopedIds ids(invlists, key);

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                nlistv++;
                ndis += list_size;
                scanner->scan_codes_range(
                        list_size, scodes.get(), ids.get(), radius, qres);

            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

        if (parallel_mode == 0) {
#pragma omp for
            for (idx_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }
        } else if (parallel_mode == 1) {
            for (size_t i = 0; i < nx; i++) { // 没声明‘#pragma omp for’，说明是串行查询每个向量
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i); // 这里是单线程运行，只会声明1个qres？

#pragma omp for schedule(dynamic) // 对各个聚类的查询做并行
                for (int64_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres); // qres 是所有线程共享？可以并行写入吗？
                }
            }
        } else if (parallel_mode == 2) {
            RangeQueryResult* qres = nullptr; // 每个线程都会声明

#pragma omp for schedule(dynamic)
            for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
                idx_t i = iik / (idx_t)nprobe;
                idx_t ik = iik % (idx_t)nprobe;
                if (qres == nullptr || qres->qno != i) {
                    qres = &pres.new_result(i);
                    scanner->set_query(x + i * d);
                }
                scan_list_func(i, ik, *qres);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", parallel_mode);
        }
        if (parallel_mode == 0) { // 并行查询共享同一个pres，用finalize合并
            pres.finalize();
        } else {
#pragma omp barrier
#pragma omp single
            RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
        }
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (stats) {
        stats->nq += nx;
        stats->nlist += nlistv;
        stats->ndis += ndis;
    }
}

InvertedListScanner* IndexIVF::get_InvertedListScanner(
        bool /*store_pairs*/) const {
    return nullptr;
}

void IndexIVF::reconstruct(idx_t key, float* recons) const {
    idx_t lo = direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

void IndexIVF::reconstruct_n(idx_t i0, idx_t ni, float* recons) const { // 找出 id 位于 [i0, i0+ni) 的向量并重构
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        ScopedIds idlist(invlists, list_no); // 会释放倒排列表的空间

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

/* standalone codec interface */
size_t IndexIVF::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    return code_size + coarse_size;
}

void IndexIVF::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    quantizer->assign(n, x, idx.get());
    encode_vectors(n, x, idx.get(), bytes, true);
}

void IndexIVF::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons) const {
    FAISS_THROW_IF_NOT(k > 0);

    const size_t nprobe = std::min(nlist, this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t* idx = new idx_t[n * nprobe];
    ScopeDeleter<idx_t> del(idx);
    float* coarse_dis = new float[n * nprobe];
    ScopeDeleter<float> del2(coarse_dis);

    quantizer->search(n, x, nprobe, coarse_dis, idx);

    invlists->prefetch_lists(idx, n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx,
            coarse_dis,
            distances,
            labels,
            true /* store_pairs */);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            float* reconstructed = recons + ij * d; // 离 i 向量第 j+1 近的向量的重构位置
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                int list_no = lo_listno(key);
                int offset = lo_offset(key);

                // Update label to the actual id
                labels[ij] = invlists->get_single_id(list_no, offset);

                reconstruct_from_offset(list_no, offset, reconstructed);
            }
        }
    }
}

void IndexIVF::reconstruct_from_offset(
        int64_t /*list_no*/,
        int64_t /*offset*/,
        float* /*recons*/) const {
    FAISS_THROW_MSG("reconstruct_from_offset not implemented");
}

void IndexIVF::reset() {
    direct_map.clear();
    invlists->reset();
    ntotal = 0;
}

size_t IndexIVF::remove_ids(const IDSelector& sel) {
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}

void IndexIVF::update_vectors(int n, const idx_t* new_ids, const float* x) {
    if (direct_map.type == DirectMap::Hashtable) { // 使用 direct_map 删除所有 id 及编码之后，再计算新的聚类，插入到新的倒排列表
        // just remove then add
        IDSelectorArray sel(n, new_ids);
        size_t nremove = remove_ids(sel);
        FAISS_THROW_IF_NOT_MSG(
                nremove == n, "did not find all entries to remove");
        add_with_ids(n, x, new_ids);
        return;
    }

    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Array);
    // here it is more tricky because we don't want to introduce holes
    // in continuous range of ids

    FAISS_THROW_IF_NOT(is_trained);
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data()); // 不使用 direct_map.remove_ids 的穷举搜索来删除

    std::vector<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, assign.data(), flat_codes.data()); // 原理是一样的，只是先算了新的聚类，然后根据 direct_map 在一个个删除旧的 id 及编码的同时，插入新编码到新的倒排列表

    direct_map.update_codes(
            invlists, n, new_ids, assign.data(), flat_codes.data());
}

void IndexIVF::train(idx_t n, const float* x) {
    if (verbose)
        printf("Training level-1 quantizer\n");

    train_q1(n, x, verbose, metric_type);

    if (verbose)
        printf("Training IVF residual\n");

    train_residual(n, x);
    is_trained = true;
}

void IndexIVF::train_residual(idx_t /*n*/, const float* /*x*/) {
    if (verbose)
        printf("IndexIVF: no residual training\n");
    // does nothing by default
}

void IndexIVF::check_compatible_for_merge(const IndexIVF& other) const {
    // minimal sanity checks
    FAISS_THROW_IF_NOT(other.d == d);
    FAISS_THROW_IF_NOT(other.nlist == nlist);
    FAISS_THROW_IF_NOT(other.code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(other), // 相同的派生类
            "can only merge indexes of the same type");
    FAISS_THROW_IF_NOT_MSG(
            this->direct_map.no() && other.direct_map.no(),
            "merge direct_map not implemented");
}

void IndexIVF::merge_from(IndexIVF& other, idx_t add_id) {
    check_compatible_for_merge(other);

    invlists->merge_from(other.invlists, add_id); // 把 other 里的 id 都增加 add_id 得到新的 id 插入 this

    ntotal += other.ntotal;
    other.ntotal = 0;
}

void IndexIVF::replace_invlists(InvertedLists* il, bool own) {
    if (own_invlists) {
        delete invlists;
        invlists = nullptr;
    }
    // FAISS_THROW_IF_NOT (ntotal == 0);
    if (il) {
        FAISS_THROW_IF_NOT(il->nlist == nlist);
        FAISS_THROW_IF_NOT(
                il->code_size == code_size ||
                il->code_size == InvertedLists::INVALID_CODE_SIZE); // BlockInvertedLists 中 codes 组合起来了，没有单个 code_size 概念
    }
    invlists = il;
    own_invlists = own;
}

void IndexIVF::copy_subset_to( // 拷贝自己的向量到别到索引上，自己的所有向量最终都会被删除、释放掉
        IndexIVF& other,
        int subset_type,
        idx_t a1,
        idx_t a2) const {
    FAISS_THROW_IF_NOT(nlist == other.nlist);
    FAISS_THROW_IF_NOT(code_size == other.code_size);
    FAISS_THROW_IF_NOT(other.direct_map.no()); // other 必须是 NoMap
    FAISS_THROW_IF_NOT_FMT(
            subset_type == 0 || subset_type == 1 || subset_type == 2,
            "subset type %d not implemented",
            subset_type);

    size_t accu_n = 0; // 记录处理的向量个数
    size_t accu_a1 = 0;
    size_t accu_a2 = 0;

    InvertedLists* oivf = other.invlists;

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = invlists->list_size(list_no);
        ScopedIds ids_in(invlists, list_no);

        if (subset_type == 0) { // copies ids in [a1, a2)
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (a1 <= id && id < a2) {
                    oivf->add_entry(
                            list_no,
                            invlists->get_single_id(list_no, i), // 这个不就是 id 吗？
                            ScopedCodes(invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 1) {
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (id % a1 == a2) {
                    oivf->add_entry(
                            list_no,
                            invlists->get_single_id(list_no, i),
                            ScopedCodes(invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 2) {
            // see what is allocated to a1 and to a2
            size_t next_accu_n = accu_n + n; // 算上当前倒排表一共处理了多少向量
            size_t next_accu_a1 = next_accu_n * a1 / ntotal; // next_accu_n 的一个小比例的值
            size_t i1 = next_accu_a1 - accu_a1; // accu_a1增加量: ((accu_n + n) * a1 / ntotal) - accu_n * a1 / ntotal = n * a1 / ntotal
            size_t next_accu_a2 = next_accu_n * a2 / ntotal; // next_accu_n 的一个大比例的值
            size_t i2 = next_accu_a2 - accu_a2; // accu_a2增加量: ((accu_n + n) * a2 / ntotal) - accu_n * a2 / ntotal = n * a2 / ntotal

            for (idx_t i = i1; i < i2; i++) { // 从每个倒排列表里选择一个固定比例即 [n * a1 / ntotal, n * a2 / ntotal) 的ID来拷贝
                oivf->add_entry(
                        list_no,
                        invlists->get_single_id(list_no, i),
                        ScopedCodes(invlists, list_no, i).get());
            }

            other.ntotal += i2 - i1;
            accu_a1 = next_accu_a1;
            accu_a2 = next_accu_a2;
        }
        accu_n += n;
    }
    FAISS_ASSERT(accu_n == ntotal);
}

IndexIVF::~IndexIVF() {
    if (own_invlists) {
        delete invlists;
    }
}

/*************************************************************************
 * IndexIVFStats
 *************************************************************************/

void IndexIVFStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

void IndexIVFStats::add(const IndexIVFStats& other) {
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    nheap_updates += other.nheap_updates;
    quantization_time += other.quantization_time;
    search_time += other.search_time;
}

IndexIVFStats indexIVF_stats; // 全局变量！

/*************************************************************************
 * InvertedListScanner
 *************************************************************************/

size_t InvertedListScanner::scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const {
    size_t nup = 0;

    if (!keep_max) {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis < simi[0]) { // 找到一个更小值
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j]; // 构造id
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    } else {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    }
    return nup;
}

void InvertedListScanner::scan_codes_range(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float radius,
        RangeQueryResult& res) const {
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }
        codes += code_size;
    }
}

} // namespace faiss
