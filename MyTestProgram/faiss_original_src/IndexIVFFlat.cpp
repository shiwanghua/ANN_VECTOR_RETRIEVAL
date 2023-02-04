/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlat.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric) {
    code_size = sizeof(float) * d;
}

void IndexIVFFlat::add_core( // 没有像父类一样对 x 进行编码了，flat 不需要编码，这里也没有存储倒排表号
        idx_t n,
        const float* x,
        const int64_t* xids,
        const int64_t* coarse_idx)

{
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    assert(invlists);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset =
                        invlists->add_entry(list_no, id, (const uint8_t*)xi);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) { // 处理无效向量
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFFlat::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (!include_listnos) { // 不存储倒排表序号到码字(默认情况)
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size(); // 存储位置需要的字节数
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else { // 表号无效，占位
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexIVFFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const { // 对 include_listnos 为 true 的情况进行解码
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

namespace {

template <MetricType metric, class C>
struct IVFFlatScanner : InvertedListScanner {
    size_t d;

    IVFFlatScanner(size_t d, bool store_pairs) : d(d) {
        this->store_pairs = store_pairs;
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
        return dis;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (C::cmp(simi[0], dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (C::cmp(radius, dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlat::get_InvertedListScanner(
        bool store_pairs) const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        return new IVFFlatScanner<METRIC_INNER_PRODUCT, CMin<float, int64_t>>(
                d, store_pairs);
    } else if (metric_type == METRIC_L2) {
        return new IVFFlatScanner<METRIC_L2, CMax<float, int64_t>>(
                d, store_pairs);
    } else {
        FAISS_THROW_MSG("metric type not supported");
    }
    return nullptr;
}

void IndexIVFFlat::reconstruct_from_offset( // 没存储位置到列表，直接取 code
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

/*****************************************
 * IndexIVFFlatDedup implementation
 ******************************************/

IndexIVFFlatDedup::IndexIVFFlatDedup(
        Index* quantizer,
        size_t d,
        size_t nlist_,
        MetricType metric_type)
        : IndexIVFFlat(quantizer, d, nlist_, metric_type) {}

void IndexIVFFlatDedup::train(idx_t n, const float* x) {
    std::unordered_map<uint64_t, idx_t> map;
    std::unique_ptr<float[]> x2(new float[n * d]);

    int64_t n2 = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t hash = hash_bytes((uint8_t*)(x + i * d), code_size);
        if (map.count(hash) &&
            !memcmp(x2.get() + map[hash] * d, x + i * d, code_size)) {
            // is duplicate, skip
        } else {
            map[hash] = n2;
            memcpy(x2.get() + n2 * d, x + i * d, code_size);
            n2++;
        }
    }
    if (verbose) {
        printf("IndexIVFFlatDedup::train: train on %" PRId64
               " points after dedup "
               "(was %" PRId64 " points)\n",
               n2,
               n);
    }
    IndexIVFFlat::train(n2, x2.get());
}

void IndexIVFFlatDedup::add_with_ids(
        idx_t na,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "IVFFlatDedup not implemented with direct_map");
    std::unique_ptr<int64_t[]> idx(new int64_t[na]);
    quantizer->assign(na, x, idx.get());

    int64_t n_add = 0, n_dup = 0;

#pragma omp parallel reduction(+ : n_add, n_dup)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < na; i++) {
            int64_t list_no = idx[i];

            if (list_no < 0 || list_no % nt != rank) { // 每个线程单独负责多个倒排列表，互不干扰，可以并行
                continue;
            }

            idx_t id = xids ? xids[i] : ntotal + i;
            const float* xi = x + i * d;

            // search if there is already an entry with that id
            InvertedLists::ScopedCodes codes(invlists, list_no);

            int64_t n = invlists->list_size(list_no);
            int64_t offset = -1;
            for (int64_t o = 0; o < n; o++) {
                if (!memcmp(codes.get() + o * code_size, xi, code_size)) {
                    offset = o;
                    break;
                }
            }

            if (offset == -1) { // not found
                invlists->add_entry(list_no, id, (const uint8_t*)xi);
            } else {
                // mark equivalence
                idx_t id2 = invlists->get_single_id(list_no, offset);
                std::pair<idx_t, idx_t> pair(id2, id); // id2是实际存储的id，id是共享存储的向量id

#pragma omp critical
                // executed by one thread at a time
                instances.insert(pair);

                n_dup++;
            }
            n_add++; // 为什么重复时没插入倒排列表中，但也加了一，最后也加到了 ntotal 上？因为要实现重复向量也插入了的效果！
        }
    }
    if (verbose) {
        printf("IndexIVFFlat::add_with_ids: added %" PRId64 " / %" PRId64
               " vectors"
               " (out of which %" PRId64 " are duplicates)\n",
               n_add,
               na,
               n_dup); // n_add==na
    }
    ntotal += n_add;
}

void IndexIVFFlatDedup::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported in IVFDedup");

    IndexIVFFlat::search_preassigned(
            n, x, k, assign, centroid_dis, distances, labels, false, params);

    std::vector<idx_t> labels2(k);
    std::vector<float> dis2(k);

    for (int64_t i = 0; i < n; i++) {
        idx_t* labels1 = labels + i * k;
        float* dis1 = distances + i * k;
        int64_t j = 0;
        for (; j < k; j++) {
            if (instances.find(labels1[j]) != instances.end()) { // 第 i 个向量的 k 个结果向量中有重复
                // a duplicate: special handling
                break;
            }
        }
        if (j < k) {
            // there are duplicates, special handling
            int64_t j0 = j;
            int64_t rp = j;
            while (j < k) { // 小于 j 的向量没重复无需处理
                auto range = instances.equal_range(labels1[rp]);
                float dis = dis1[rp]; // 多个相同的向量，到查询向量的距离是一样的
                labels2[j] = labels1[rp]; // 相同的向量的第一个实例，id 不变
                dis2[j] = dis;
                j++;
                for (auto it = range.first; j < k && it != range.second; ++it) {
                    labels2[j] = it->second; // 遍历其他相同向量的 id 作为查询结果
                    dis2[j] = dis;
                    j++;
                }
                rp++; // 从原始查询结果中处理找下一个最近的向量
            }
            memcpy(labels1 + j0,
                   labels2.data() + j0,
                   sizeof(labels1[0]) * (k - j0)); // 前面 j0(+1) 个保留
            memcpy(dis1 + j0, dis2.data() + j0, sizeof(dis2[0]) * (k - j0));
        }
    }
}

size_t IndexIVFFlatDedup::remove_ids(const IDSelector& sel) {
    std::unordered_map<idx_t, idx_t> replace;
    std::vector<std::pair<idx_t, idx_t>> toadd;
    for (auto it = instances.begin(); it != instances.end();) { // 每个元素是一个pair
        if (sel.is_member(it->first)) {
            // then we erase this entry
            if (!sel.is_member(it->second)) {
                // if the second is not erased
                if (replace.count(it->first) == 0) {
                    replace[it->first] = it->second; // 这是 first 对应的第一个 second，在倒排列上把 first 替换为 second (在后面omp里执行)
                } else { // remember we should add an element
                    std::pair<idx_t, idx_t> new_entry( // 原来的 second 已经占了同一个 first 的位置，后来的 second 只能存到 multimap 上了
                            replace[it->first], it->second);
                    toadd.push_back(new_entry);
                }
            }
            it = instances.erase(it); // 把 first 和 second 都删除了，如果 second 不需删除， 其信息保留在了 replace 或 toadd 中
        } else {
            if (sel.is_member(it->second)) {
                it = instances.erase(it); // 返回下一个迭代器，如果这条 pair 是最后一个了，first 就是唯一的了，不重复了，也应该在multimap里删除
            } else {
                ++it;
            }
        }
    }

    instances.insert(toadd.begin(), toadd.end());

    // mostly copied from IndexIVF.cpp

    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "direct map remove not implemented");

    std::vector<int64_t> toremove(nlist);

#pragma omp parallel for
    for (int64_t i = 0; i < nlist; i++) {
        int64_t l0 = invlists->list_size(i), l = l0, j = 0;
        InvertedLists::ScopedIds idsi(invlists, i); // 遍历每个倒排列表的每个向量id
        while (j < l) {
            if (sel.is_member(idsi[j])) {
                if (replace.count(idsi[j]) == 0) { // 这个 id 没有重复向量，直接用列表最后一个向量覆盖
                    l--;
                    invlists->update_entry(
                            i,
                            j,
                            invlists->get_single_id(i, l),
                            InvertedLists::ScopedCodes(invlists, i, l).get()); // 覆盖 j 偏移位置的向量
                } else { // 这个 id 还有重复向量，其中第一个重复向量的 id 为 replace[idsi[j]]
                    invlists->update_entry(
                            i,
                            j,
                            replace[idsi[j]],
                            InvertedLists::ScopedCodes(invlists, i, j).get());
                    j++;
                }
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l; // 倒排列表上实际删除了 l0-l 个向量
    }
    // this will not run well in parallel on ondisk because of possible shrinks
    int64_t nremove = 0;
    for (int64_t i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove; // 没有减去删除重复向量的个数！但前面插入时加上了重复向量的个数到 ntotal！？
    return nremove;
}

void IndexIVFFlatDedup::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*) const {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::update_vectors(int, const idx_t*, const float*) {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::reconstruct_from_offset(int64_t, int64_t, float*)
        const {
    FAISS_THROW_MSG("not implemented");
}

} // namespace faiss
