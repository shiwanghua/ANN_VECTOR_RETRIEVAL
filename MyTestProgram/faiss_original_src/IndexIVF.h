/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_H
#define FAISS_INDEX_IVF_H

#include <stdint.h>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

namespace faiss {

/** Encapsulates a quantizer object for the IndexIVF
 *
 * The class isolates the fields that are independent of the storage
 * of the lists (especially training)
 */
struct Level1Quantizer {
    // 存储nlist个质心向量
    Index* quantizer; ///< quantizer that maps vectors to inverted lists
    size_t nlist;     ///< number of possible key values

    /**
     * = 0: use the quantizer as index in a kmeans training
     * = 1: just pass on the training set to the train() of the quantizer
     * = 2: kmeans training on a flat index + add the centroids to the quantizer
     */
    char quantizer_trains_alone;
    bool own_fields; ///< whether object owns the quantizer (false by default)

    ClusteringParameters cp; ///< to override default clustering params
    Index* clustering_index; ///< to override index used during clustering 聚类训练时存储质心、进行搜索的索引

    /// Trains the quantizer and calls train_residual to train sub-quantizers // 这个函数里不会调用 train_residual
    void train_q1(
            size_t n,
            const float* x,
            bool verbose,
            MetricType metric_type);

    /// compute the number of bytes required to store list ids
    size_t coarse_code_size() const;
    void encode_listno(Index::idx_t list_no, uint8_t* code) const; // 对倒排表号进行编码...
    Index::idx_t decode_listno(const uint8_t* code) const;

    Level1Quantizer(Index* quantizer, size_t nlist);

    Level1Quantizer();

    ~Level1Quantizer();
};

struct IVFSearchParameters {
    size_t nprobe;    ///< number of probes at query time
    size_t max_codes; ///< max nb of codes to visit to do a query 精确搜索向量个数的上限
    IVFSearchParameters() : nprobe(1), max_codes(0) {}
    virtual ~IVFSearchParameters() {}
};

struct InvertedListScanner;
struct IndexIVFStats;

/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an Index instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is stored.
 *
 * The inverted list object is required only after trainng. If none is
 * set externally, an ArrayInvertedLists is used automatically.
 *
 * At search time, the vector to be searched is also quantized, and
 * only the list corresponding to the quantization index is
 * searched. This speeds up the search by making it
 * non-exhaustive. This can be relaxed using multi-probe search: a few
 * (nprobe) quantization indices are selected and several inverted
 * lists are visited.
 *
 * Sub-classes implement a post-filtering of the index that refines
 * the distance estimation from the query to databse vectors.
 */
struct IndexIVF : Index, Level1Quantizer {
    /// Access to the actual data
    InvertedLists* invlists;
    bool own_invlists;

    size_t code_size; ///< code size per vector in bytes

    size_t nprobe;    ///< number of probes at query time
    size_t max_codes; ///< max nb of codes to visit to do a query

    /** Parallel mode determines how queries are parallelized with OpenMP
     *
     * 0 (default): split over queries // 不并行
     * 1: parallelize over inverted lists
     * 2: parallelize over both
     * 3: split over queries with a finer granularity // 范围搜索时不能等于3
     *
     * PARALLEL_MODE_NO_HEAP_INIT: binary or with the previous to
     * prevent the heap to be initialized and finalized
     */
    int parallel_mode;
    const int PARALLEL_MODE_NO_HEAP_INIT = 1024;

    /** optional map that maps back ids to invlist entries. This
     *  enables reconstruct() */
    DirectMap direct_map;

    /** The Inverted file takes a quantizer (an Index) on input,
     * which implements the function mapping a vector to a list
     * identifier.
     */
    IndexIVF(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2);

    void reset() override;

    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train(idx_t n, const float* x) override;

    /// Calls add_with_ids with NULL ids
    void add(idx_t n, const float* x) override;

    /// default implementation that calls add_core
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    /** Implementation of vector addition where the vector assignments are
     * predefined. The default implementation hands over the code extraction to
     * encode_vectors.
     *
     * @param precomputed_idx    quantization indices for the input vectors
     * (size n)
     */
    virtual void add_core( // add->add_with_ids 会调用，把向量插入到各自所属的倒排列表
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx);

    /** Encodes a set of vectors as they would appear in the inverted lists
     *
     * @param list_nos   inverted list ids as returned by the // 所属的倒排列表号
     *                   quantizer (size n). -1s are ignored. // -1 表示这个向量无效，不需要编码
     * @param codes      output codes, size n * code_size // 应该是 n * sa_code_size() ？
     * @param include_listno // 如果为true，倒排列表号也会按字节存进code里
     *                   include the list ids in the code (in this case add
     *                   ceil(log8(nlist)) to the code size) // 为什么是以 8 为底而不是 256 为底？
     */
    virtual void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const = 0;

    /** Add vectors that are computed with the standalone codec
     *
     * @param codes  codes to add size n * sa_code_size()
     * @param xids   corresponding ids, size n
     */
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids);

    /// Sub-classes that encode the residuals can train their encoders here
    /// does nothing by default
    virtual void train_residual(idx_t n, const float* x);

    /** search a set of vectors, that are pre-quantized by the IVF
     *  quantizer. Fill in the corresponding heaps with the query
     *  results. The default implementation uses InvertedListScanners
     *  to do the search.
     *
     * @param n      nb of vectors to query
     * @param x      query vectors, size nx * d
     * @param assign coarse quantization indices, size nx * nprobe // 每个查询向量有 nprobe 个需要详细检查的类
     * @param centroid_dis
     *               distances to coarse centroids, size nx * nprobe
     * @param distance
     *               output distances, size n * k
     * @param labels output labels, size n * k
     * @param store_pairs store inv list index + inv list offset
     *                     instead in upper/lower 32 bit of result,
     *                     instead of ids (used for reranking).
     * @param params used to override the object's search parameters
     * @param stats  search stats to be updated (can be null)
     */
    virtual void search_preassigned( // 相当于“二级检索”
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const;

    /** assign the vectors, then call search_preassign */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result) const override;

    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const;

    /** Get a scanner for this index (store_pairs means ignore labels)
     *
     * The default search implementation uses this to compute the distances
     */
    virtual InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false) const; // 默认不存储列表号和偏移，而是直接用id为结果

    /** reconstruct a vector. Works only if maintain_direct_map is set to 1 or 2
     */
    void reconstruct(idx_t key, float* recons) const override;

    /** Update a subset of vectors.
     *
     * The index must have a direct_map
     *
     * @param nv     nb of vectors to update
     * @param idx    vector indices to update, size nv
     * @param v      vectors of new values, size nv*d
     */
    virtual void update_vectors(int nv, const idx_t* idx, const float* v);

    /** Reconstruct a subset of the indexed vectors.
     *
     * Overrides default implementation to bypass reconstruct() which requires
     * direct_map to be maintained.        // 绕开 reconstruct
     *
     * @param i0     first vector to reconstruct
     * @param ni     nb of vectors to reconstruct
     * @param recons output array of reconstructed vectors, size ni * d
     */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /** Similar to search, but also reconstructs the stored vectors (or an
     * approximation in the case of lossy coding) for the search results.
     *
     * Overrides default implementation to avoid having to maintain direct_map
     * and instead fetch the code offsets through the `store_pairs` flag in
     * search_preassigned().
     *
     * @param recons      reconstructed vectors size (n, k, d)
     */
    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* recons) const override;

    /** Reconstruct a vector given the location in terms of (inv list index +
     * inv list offset) instead of the id.
     *
     * Useful for reconstructing when the direct_map is not maintained and
     * the inv list offset is computed by search_preassigned() with
     * `store_pairs` set.
     */
    virtual void reconstruct_from_offset(
            int64_t list_no,
            int64_t offset,
            float* recons) const;

    /// Dataset manipulation functions

    size_t remove_ids(const IDSelector& sel) override;

    /** check that the two indexes are compatible (ie, they are
     * trained in the same way and have the same
     * parameters). Otherwise throw. */
    void check_compatible_for_merge(const IndexIVF& other) const;

    /** moves the entries from another dataset to self. On output,
     * other is empty. add_id is added to all moved ids (for
     * sequential ids, this would be this->ntotal */
    virtual void merge_from(IndexIVF& other, idx_t add_id); // other 里的向量 id 都加上 add_id, 如果是顺序添加到末尾，则 add_id=ntotal

    /** copy a subset of the entries index to the other index
     *
     * if subset_type == 0: copies ids in [a1, a2) // 把 id 位于区间里的向量拷贝过去
     * if subset_type == 1: copies ids if id % a1 == a2
     * if subset_type == 2: copies inverted lists such that a1
     *                      elements are left before and a2 elements are after // 从每个倒排列表里选择一个固定比例即 [n * a1 / ntotal, n * a2 / ntotal) 的ID来拷贝
     */
    virtual void copy_subset_to( // 拷贝自己的向量到别到索引上，自己的所有向量最终都会被删除、释放掉
            IndexIVF& other,
            int subset_type,
            idx_t a1,
            idx_t a2) const;

    ~IndexIVF() override;

    size_t get_list_size(size_t list_no) const {
        return invlists->list_size(list_no);
    }

    /** intialize a direct map
     *
     * @param new_maintain_direct_map    if true, create a direct map,
     *                                   else clear it
     */
    void make_direct_map(bool new_maintain_direct_map = true);

    void set_direct_map_type(DirectMap::Type type);

    /// replace the inverted lists, old one is deallocated if own_invlists
    void replace_invlists(InvertedLists* il, bool own = false);

    /* The standalone codec interface (except sa_decode that is specific) */
    size_t sa_code_size() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    IndexIVF();
};

struct RangeQueryResult;

/** Object that handles a query. The inverted lists to scan are
 * provided externally. The object has a lot of state, but
 * distance_to_code and scan_codes can be called in multiple
 * threads */
struct InvertedListScanner { // 没有存储查询向量
    using idx_t = Index::idx_t;

    idx_t list_no = -1;    ///< remember current list
    bool keep_max = false; ///< keep maximum instead of minimum
    /// store positions in invlists rather than labels
    bool store_pairs = false;

    /// used in default implementation of scan_codes
    size_t code_size = 0;

    /// from now on we handle this query.
    virtual void set_query(const float* query_vector) = 0;

    /// following codes come from this inverted list
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;

    /// compute a single query-to-code distance
    virtual float distance_to_code(const uint8_t* code) const = 0;

    /** scan a set of codes, compute distances to current query and
     * update heap of results if necessary. Default implemetation
     * calls distance_to_code.
     *
     * @param n      number of codes to scan
     * @param codes  codes to scan (n * code_size)
     * @param ids        corresponding ids (ignored if store_pairs)
     * @param distances  heap distances (size k)
     * @param labels     heap labels (size k)
     * @param k          heap size
     * @return number of heap updates performed
     */
    virtual size_t scan_codes( // 只有一个查询向量的搜索函数
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const;

    /** scan a set of codes, compute distances to current query and
     * update results if distances are below radius
     *
     * (default implementation fails) */
    virtual void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result) const;

    virtual ~InvertedListScanner() {}
};

struct IndexIVFStats {
    size_t nq;                // nb of queries run
    size_t nlist;             // nb of inverted lists scanned
    size_t ndis;              // nb of distances computed
    size_t nheap_updates;     // nb of times the heap was updated
    double quantization_time; // time spent quantizing vectors (in ms) 一级搜索的时间
    double search_time;       // time spent searching lists (in ms) 二级搜索时间

    IndexIVFStats() {
        reset();
    }
    void reset();
    void add(const IndexIVFStats& other);
};

// global var that collects them all
FAISS_API extern IndexIVFStats indexIVF_stats;

} // namespace faiss

#endif