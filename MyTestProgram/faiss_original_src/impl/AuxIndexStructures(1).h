/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef FAISS_AUX_INDEX_STRUCTURES_H
#define FAISS_AUX_INDEX_STRUCTURES_H

#include <stdint.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  do_allocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
struct RangeSearchResult { // 存储所有查询向量的所有查询结果
    size_t nq;    ///< nb of queries
    size_t* lims; ///< size (nq + 1) 前缀和（每个查询找到了多少个向量，这些数构成的数组的前缀和）

    typedef Index::idx_t idx_t;

    idx_t* labels;    ///< result for query i is labels[lims[i]:lims[i+1]]  所有查询结果都是连在一起的，通过lims确定边界
    float* distances; ///< corresponding distances (not sorted)

    size_t buffer_size; ///< size of the result buffers used

    /// lims must be allocated on input to range_search.
    explicit RangeSearchResult(idx_t nq, bool alloc_lims = true);

    /// called when lims contains the nb of elements result entries
    /// for each query

    virtual void do_allocation();

    virtual ~RangeSearchResult();
};

/** Encapsulates a set of ids to remove. */
struct IDSelector {
    typedef Index::idx_t idx_t;
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};

/** remove ids between [imni, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    IDSelectorRange(idx_t imin, idx_t imax);
    bool is_member(idx_t id) const override;
    ~IDSelectorRange() override {}
};

/** simple list of elements to remove
 *
 * this is inefficient in most cases, except for IndexIVF with
 * maintain_direct_map
 */
struct IDSelectorArray : IDSelector {
    size_t n;
    const idx_t* ids;

    IDSelectorArray(size_t n, const idx_t* ids);
    bool is_member(idx_t id) const override;
    ~IDSelectorArray() override {}
};

/** Remove ids from a set. Repetitions of ids in the indices set
 * passed to the constructor does not hurt performance. The hash
 * function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This
 * works fine for random ids or ids in sequences but will produce many
 * hash collisions if lsb's are always the same */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    typedef unsigned char uint8_t;
    std::vector<uint8_t> bloom; // assumes low bits of id are a good hash value
    int nbits;
    idx_t mask;

    IDSelectorBatch(size_t n, const idx_t* indices);
    bool is_member(idx_t id) const override;
    ~IDSelectorBatch() override {}
};

/****************************************************************
 * Result structures for range search.
 *
 * The main constraint here is that we want to support parallel
 * queries from different threads in various ways: 1 thread per query,
 * several threads per query. We store the actual results in blocks of
 * fixed size rather than exponentially increasing memory. At the end,
 * we copy the block content to a linear result array.
 *****************************************************************/

/** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
struct BufferList {
    typedef Index::idx_t idx_t;

    // buffer sizes in # entries
    size_t buffer_size; // 每个buffer里最多可存储这么多个id和dis

    struct Buffer {
        idx_t* ids;
        float* dis;
    };

    std::vector<Buffer> buffers;
    size_t wp; ///< write pointer in the last buffer.

    explicit BufferList(size_t buffer_size);

    ~BufferList();

    /// create a new buffer
    void append_buffer();

    /// add one result, possibly appending a new buffer if needed
    void add(idx_t id, float dis);

    /// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
    /// tables dest_ids, dest_dis
    void copy_range(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
};

struct RangeSearchPartialResult;

/// result structure for a single query
struct RangeQueryResult {
    using idx_t = Index::idx_t;
    idx_t qno;   //< id of the query
    size_t nres; //< nb of results for this query 这个查询的结果个数
    RangeSearchPartialResult* pres;

    /// called by search function to report a new result
    void add(float dis, idx_t id);
};

/// the entries in the buffers are split per query
struct RangeSearchPartialResult : BufferList { // 存储多个查询向量的完整结果，Partial 指多线程时每个线程的 RangeSearchPartialResult 是部分的
    RangeSearchResult* res; // 存储这个线程的最终查询结果

    /// eventually the result will be stored in res_in
    explicit RangeSearchPartialResult(RangeSearchResult* res_in);

    /// query ids + nb of results per query.
    std::vector<RangeQueryResult> queries; // 相互引用，queries的每个元素存储一个查询向量的查询结果

    /// begin a new result
    RangeQueryResult& new_result(idx_t qno);

    /*****************************************
     * functions used at the end of the search to merge the result
     * lists */
    void finalize();

    /// called by range_search before do_allocation
    void set_lims();

    /// called by range_search after do_allocation
    void copy_result(bool incremental = false); // 合并1个线程负责的多个查询向量的查询结果到res中

    /// merge a set of PartialResult's into one RangeSearchResult
    /// on ouptut the partialresults are empty!
    static void merge(
            std::vector<RangeSearchPartialResult*>& partial_results,
            bool do_delete = true);
};

/***********************************************************
 * Interrupt callback
 ***********************************************************/

struct FAISS_API InterruptCallback {
    virtual bool want_interrupt() = 0;
    virtual ~InterruptCallback() {}

    // lock that protects concurrent calls to is_interrupted
    static std::mutex lock;

    static std::unique_ptr<InterruptCallback> instance;

    static void clear_instance();

    /** check if:
     * - an interrupt callback is set
     * - the callback returns true
     * if this is the case, then throw an exception. Should not be called
     * from multiple threads.
     */
    static void check();

    /// same as check() but return true if is interrupted instead of
    /// throwing. Can be called from multiple threads.
    static bool is_interrupted();

    /** assuming each iteration takes a certain number of flops, what
     * is a reasonable interval to check for interrupts?
     */
    static size_t get_period_hint(size_t flops);
};

/// set implementation optimized for fast access.
struct VisitedTable {
    std::vector<uint8_t> visited;
    int visno;

    explicit VisitedTable(int size) : visited(size), visno(1) {}

    /// set flag #no to true
    void set(int no) {
        visited[no] = visno;
    }

    /// get flag #no
    bool get(int no) const {
        return visited[no] == visno;
    }

    /// reset all flags to false
    void advance() {
        visno++;
        if (visno == 250) {
            // 250 rather than 255 because sometimes we use visno and visno+1 // when? // 254?
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};

} // namespace faiss

#endif
