/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <algorithm>
#include <cstring>

#include <faiss/impl/AuxIndexStructures.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

/***********************************************************************
 * RangeSearchResult
 ***********************************************************************/

RangeSearchResult::RangeSearchResult(idx_t nq, bool alloc_lims) : nq(nq) {
    if (alloc_lims) {
        lims = new size_t[nq + 1];
        memset(lims, 0, sizeof(*lims) * (nq + 1));
    } else {
        lims = nullptr;
    }
    labels = nullptr;
    distances = nullptr;
    buffer_size = 1024 * 256;
}

/// called when lims contains the nb of elements result entries
/// for each query
void RangeSearchResult::do_allocation() {
    // works only if all the partial results are aggregated
    // simulatenously
    FAISS_THROW_IF_NOT(labels == nullptr && distances == nullptr);
    size_t ofs = 0;
    for (int i = 0; i < nq; i++) {
        size_t n = lims[i];
        lims[i] = ofs; // lims 在 0 到 i-1 号查询向量的查询结果个数的和
        ofs += n;
    }
    lims[nq] = ofs;
    labels = new idx_t[ofs];
    distances = new float[ofs];
}

RangeSearchResult::~RangeSearchResult() {
    delete[] labels;
    delete[] distances;
    delete[] lims;
}

/***********************************************************************
 * BufferList
 ***********************************************************************/

BufferList::BufferList(size_t buffer_size) : buffer_size(buffer_size) {
    wp = buffer_size;
}

BufferList::~BufferList() {
    for (int i = 0; i < buffers.size(); i++) {
        delete[] buffers[i].ids;
        delete[] buffers[i].dis;
    }
}

void BufferList::add(idx_t id, float dis) {
    if (wp == buffer_size) { // need new buffer
        append_buffer();
    }
    Buffer& buf = buffers.back();
    buf.ids[wp] = id;
    buf.dis[wp] = dis;
    wp++;
}

void BufferList::append_buffer() {
    Buffer buf = {new idx_t[buffer_size], new float[buffer_size]};
    buffers.push_back(buf);
    wp = 0;
}

/// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
/// tables dest_ids, dest_dis
void BufferList::copy_range(
        size_t ofs, // 线程内部查询结果的偏移量（当前查询向量在 buffer 中存储的偏移量）
        size_t n,
        idx_t* dest_ids,
        float* dest_dis) {
    size_t bno = ofs / buffer_size;
    ofs -= bno * buffer_size; // 在bno buffer内的偏移量
    while (n > 0) {
        size_t ncopy = ofs + n < buffer_size ? n : buffer_size - ofs; // 该buffer内需要拷贝多少个元素
        Buffer buf = buffers[bno];
        memcpy(dest_ids, buf.ids + ofs, ncopy * sizeof(*dest_ids));
        memcpy(dest_dis, buf.dis + ofs, ncopy * sizeof(*dest_dis));
        dest_ids += ncopy;
        dest_dis += ncopy;
        ofs = 0;
        bno++;
        n -= ncopy;
    }
}

/***********************************************************************
 * RangeSearchPartialResult
 ***********************************************************************/

void RangeQueryResult::add(float dis, idx_t id) {
    nres++;
    pres->add(id, dis); // 放到 buffer 里
}

RangeSearchPartialResult::RangeSearchPartialResult(RangeSearchResult* res_in)
        : BufferList(res_in->buffer_size), res(res_in) {}

/// begin a new result
RangeQueryResult& RangeSearchPartialResult::new_result(idx_t qno) {
    RangeQueryResult qres = {qno, 0, this}; // 初始为0个查询结果向量
    queries.push_back(qres);
    return queries.back();
}

void RangeSearchPartialResult::finalize() { // 多线程并行处理多个查询结果，析构、合并结果时需要保证同步
    set_lims(); // 只是给lims赋值，还没求前缀和
#pragma omp barrier // 保证所有查询结果个数都已赋值到 lims 数组

#pragma omp single
    res->do_allocation(); // 求前缀和，统一分配空间，只需分配一次

#pragma omp barrier // 分配完毕，并行执行copy合并每个线程的部分查询结果
    copy_result();
}

/// called by range_search before do_allocation
void RangeSearchPartialResult::set_lims() { // 所有线程修改的是不同数据（自己负责的那些查询向量的查询结果个数），并行调用该函数
    for (int i = 0; i < queries.size(); i++) {
        RangeQueryResult& qres = queries[i];
        res->lims[qres.qno] = qres.nres;
    }
}

/// called by range_search after do_allocation
void RangeSearchPartialResult::copy_result(bool incremental) {
    size_t ofs = 0;
    for (int i = 0; i < queries.size(); i++) { 
        RangeQueryResult& qres = queries[i]; // 每个线程负责的查询向量不一定是连续的
        // add 到 buffer 时是直接放到了 buffer 的末尾
        // 这里都是单线程负责处理多个查询向量的查询结果
        copy_range(
                ofs, // 这不应该等于 res->lims[qres.qno] 吗？不，ofs 是针对线程内部所处理的多个查询向量而言的一个偏移量、前缀和
                qres.nres, // 这个查询向量找到了 nres 个邻近向量
                res->labels + res->lims[qres.qno], // 全局的前缀和
                res->distances + res->lims[qres.qno]);
        if (incremental) { // 默认 false
            res->lims[qres.qno] += qres.nres; // 加了后就是最后一个结果向量的后一个位置了，在 blas 版本中会这么做
        }
        ofs += qres.nres; // 又求一次前缀和
    }
}

void RangeSearchPartialResult::merge( // 用于 blas 中的归并；IndexIVF 会调用，用于合并多个线程的部分结果，每个线程可能负责多个查询的部分聚类的结果
        std::vector<RangeSearchPartialResult*>& partial_results,
        bool do_delete) {
    int npres = partial_results.size();
    if (npres == 0)
        return;
    RangeSearchResult* result = partial_results[0]->res; // 所有元素都指向同一个 RangeSearchResult
    size_t nx = result->nq; // 查询个数

    // count
    for (const RangeSearchPartialResult* pres : partial_results) {
        if (!pres)
            continue;
        for (const RangeQueryResult& qres : pres->queries) {
            result->lims[qres.qno] += qres.nres; // reduce 求和
        }
    }
    result->do_allocation(); // 统一分配存储最终结果的空间
    for (int j = 0; j < npres; j++) {
        if (!partial_results[j])
            continue;
        partial_results[j]->copy_result(true); // 合并，每个部分结果
        if (do_delete) {
            delete partial_results[j];
            partial_results[j] = nullptr;
        }
    }

    // reset the limits
    for (size_t i = nx; i > 0; i--) { // 后移一个
        result->lims[i] = result->lims[i - 1];
    }
    result->lims[0] = 0;
}

/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax)
        : imin(imin), imax(imax) {}

bool IDSelectorRange::is_member(idx_t id) const {
    return id >= imin && id < imax;
}

/***********************************************************************
 * IDSelectorArray
 ***********************************************************************/

IDSelectorArray::IDSelectorArray(size_t n, const idx_t* ids) : n(n), ids(ids) {}

bool IDSelectorArray::is_member(idx_t id) const {
    for (idx_t i = 0; i < n; i++) {
        if (ids[i] == id)
            return true;
    }
    return false;
}

/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t* indices) {
    nbits = 0;
    while (n > (1L << nbits))
        nbits++;
    nbits += 5; // 大于 nbits 时才可能映射到同一位
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = (1L << nbits) - 1;
    bloom.resize(1UL << (nbits - 3), 0); // -3是因为每个元素是一个字节，占3个二进制位
    for (long i = 0; i < n; i++) {
        Index::idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7); // 每个 id 都唯一对应 bloom 上的某一位（多对一）
    }
}

bool IDSelectorBatch::is_member(idx_t i) const {
    long im = i & mask;
    if (!(bloom[im >> 3] & (1 << (im & 7)))) { // 快速查找，如果没有可以尽快找出...
        return 0;
    }
    return set.count(i);
}

/***********************************************************
 * Interrupt callback
 ***********************************************************/

std::unique_ptr<InterruptCallback> InterruptCallback::instance;

std::mutex InterruptCallback::lock;

void InterruptCallback::clear_instance() {
    delete instance.release();
}

void InterruptCallback::check() {
    if (!instance.get()) {
        return;
    }
    if (instance->want_interrupt()) {
        FAISS_THROW_MSG("computation interrupted");
    }
}

bool InterruptCallback::is_interrupted() {
    if (!instance.get()) {
        return false;
    }
    std::lock_guard<std::mutex> guard(lock);
    return instance->want_interrupt();
}

size_t InterruptCallback::get_period_hint(size_t flops) {
    if (!instance.get()) {
        return 1L << 30; // never check
    }
    // for 10M flops, it is reasonable to check once every 10 iterations
    return std::max((size_t)10 * 10 * 1000 * 1000 / (flops + 1), (size_t)1);
}

} // namespace faiss
