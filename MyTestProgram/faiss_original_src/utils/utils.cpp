/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/utils.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <sys/types.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <sys/time.h>
#include <unistd.h>
#endif // !_MSC_VER

#include <omp.h>

#include <algorithm>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/random.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

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

/* Lapack functions, see http://www.netlib.org/clapack/old/single/sgeqrf.c */

int sgeqrf_(
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sorgqr_(
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sgemv_(
        const char* trans,
        FINTEGER* m,
        FINTEGER* n,
        float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* x,
        FINTEGER* incx,
        float* beta,
        float* y,
        FINTEGER* incy);
}

/**************************************************
 * Get some stats about the system
 **************************************************/

namespace faiss {

std::string get_compile_options() {
    std::string options;

    // this flag is set by GCC and Clang
#ifdef __OPTIMIZE__
    options += "OPTIMIZE ";
#endif

#ifdef __AVX2__
    options += "AVX2";
#elif defined(__aarch64__)
    options += "NEON";
#else
    options += "GENERIC";
#endif

    return options;
}

#ifdef _MSC_VER
double getmillisecs() {
    LARGE_INTEGER ts;
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ts);

    return (ts.QuadPart * 1e3) / freq.QuadPart;
}
#else  // _MSC_VER
double getmillisecs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
#endif // _MSC_VER

uint64_t get_cycles() {
#ifdef __x86_64__
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#ifdef __linux__

size_t get_mem_usage_kb() {
    int pid = getpid();
    char fname[256];
    snprintf(fname, 256, "/proc/%d/status", pid);
    FILE* f = fopen(fname, "r");
    FAISS_THROW_IF_NOT_MSG(f, "cannot open proc status file");
    size_t sz = 0;
    for (;;) {
        char buf[256];
        if (!fgets(buf, 256, f))
            break;
        if (sscanf(buf, "VmRSS: %ld kB", &sz) == 1)
            break;
    }
    fclose(f);
    return sz;
}

#else

size_t get_mem_usage_kb() {
    fprintf(stderr,
            "WARN: get_mem_usage_kb not implemented on current architecture\n");
    return 0;
}

#endif

void reflection(
        const float* __restrict u,
        float* __restrict x,
        size_t n,
        size_t d,
        size_t nu) {
    size_t i, j, l;
    for (i = 0; i < n; i++) {
        const float* up = u;
        for (l = 0; l < nu; l++) {
            float ip1 = 0, ip2 = 0;

            for (j = 0; j < d; j += 2) {
                ip1 += up[j] * x[j];
                ip2 += up[j + 1] * x[j + 1];
            }
            float ip = 2 * (ip1 + ip2);

            for (j = 0; j < d; j++)
                x[j] -= ip * up[j];
            up += d;
        }
        x += d;
    }
}

/* Reference implementation (slower) */
void reflection_ref(const float* u, float* x, size_t n, size_t d, size_t nu) {
    size_t i, j, l;
    for (i = 0; i < n; i++) {
        const float* up = u;
        for (l = 0; l < nu; l++) {
            double ip = 0;

            for (j = 0; j < d; j++)
                ip += up[j] * x[j];
            ip *= 2;

            for (j = 0; j < d; j++)
                x[j] -= ip * up[j];

            up += d;
        }
        x += d;
    }
}

/***************************************************************************
 * Some matrix manipulation functions
 ***************************************************************************/

void matrix_qr(int m, int n, float* a) {
    FAISS_THROW_IF_NOT(m >= n);
    FINTEGER mi = m, ni = n, ki = mi < ni ? mi : ni;
    std::vector<float> tau(ki);
    FINTEGER lwork = -1, info;
    float work_size;

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), &work_size, &lwork, &info);
    lwork = size_t(work_size);
    std::vector<float> work(lwork);

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), work.data(), &lwork, &info);

    sorgqr_(&mi, &ni, &ki, a, &mi, tau.data(), work.data(), &lwork, &info);
}

/***************************************************************************
 * Result list routines
 ***************************************************************************/

void ranklist_handle_ties(int k, int64_t* idx, const float* dis) {
    float prev_dis = -1e38;
    int prev_i = -1;
    for (int i = 0; i < k; i++) {
        if (dis[i] != prev_dis) {
            if (i > prev_i + 1) {
                // sort between prev_i and i - 1
                std::sort(idx + prev_i, idx + i);
            }
            prev_i = i;
            prev_dis = dis[i];
        }
    }
}

size_t merge_result_table_with(
        size_t n,
        size_t k,
        int64_t* I0,
        float* D0,
        const int64_t* I1,
        const float* D1,
        bool keep_min,
        int64_t translation) {
    size_t n1 = 0;

#pragma omp parallel reduction(+ : n1)
    {
        std::vector<int64_t> tmpI(k);
        std::vector<float> tmpD(k);

#pragma omp for
        for (int64_t i = 0; i < n; i++) {
            int64_t* lI0 = I0 + i * k;
            float* lD0 = D0 + i * k;
            const int64_t* lI1 = I1 + i * k;
            const float* lD1 = D1 + i * k;
            size_t r0 = 0;
            size_t r1 = 0;

            if (keep_min) {
                for (size_t j = 0; j < k; j++) {
                    if (lI0[r0] >= 0 && lD0[r0] < lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            } else {
                for (size_t j = 0; j < k; j++) {
                    if (lI0[r0] >= 0 && lD0[r0] > lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            }
            n1 += r1;
            memcpy(lD0, tmpD.data(), sizeof(lD0[0]) * k);
            memcpy(lI0, tmpI.data(), sizeof(lI0[0]) * k);
        }
    }

    return n1;
}

size_t ranklist_intersection_size( // 计算有多少个数出现在两个数组里
        size_t k1,
        const int64_t* v1,
        size_t k2,
        const int64_t* v2_in) {
    if (k2 > k1)
        return ranklist_intersection_size(k2, v2_in, k1, v1);
    int64_t* v2 = new int64_t[k2];
    memcpy(v2, v2_in, sizeof(int64_t) * k2);
    std::sort(v2, v2 + k2);
    { // de-dup v2 去重
        int64_t prev = -1;
        size_t wp = 0;
        for (size_t i = 0; i < k2; i++) {
            if (v2[i] != prev) {
                v2[wp++] = prev = v2[i];
            }
        }
        k2 = wp;
    }
    const int64_t seen_flag = int64_t{1} << 60; // 把第60位作为访问位(visit)
    size_t count = 0;
    for (size_t i = 0; i < k1; i++) {
        int64_t q = v1[i];
        size_t i0 = 0, i1 = k2;
        while (i0 + 1 < i1) {
            size_t imed = (i1 + i0) / 2;
            int64_t piv = v2[imed] & ~seen_flag; // 把第60位置0
            if (piv <= q)
                i0 = imed;
            else
                i1 = imed; // -1?
        }
        if (v2[i0] == q) { // 如果v2[i0]已经是被计数过的数了，就不可能等于q了！实现了避免重复统计的功能。
            count++;
            v2[i0] |= seen_flag; // 把第60位置1，表示v1里已经有一样的数
        }
    }
    delete[] v2;

    return count;
}

double imbalance_factor(int k, const int* hist) {
    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}

double imbalance_factor(int n, int k, const int64_t* assign) { // assign 记录每个向量属于哪个类
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++) {
        hist[assign[i]]++; // 直方图
    }

    return imbalance_factor(k, hist.data());
}

int ivec_hist(size_t n, const int* v, int vmax, int* hist) {
    memset(hist, 0, sizeof(hist[0]) * vmax);
    int nout = 0;
    while (n--) {
        if (v[n] < 0 || v[n] >= vmax)
            nout++;
        else
            hist[v[n]]++;
    }
    return nout;
}

void bincode_hist(size_t n, size_t nbits, const uint8_t* codes, int* hist) {
    FAISS_THROW_IF_NOT(nbits % 8 == 0);
    size_t d = nbits / 8;
    std::vector<int> accu(d * 256);
    const uint8_t* c = codes;
    for (size_t i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            accu[j * 256 + *c++]++;
    memset(hist, 0, sizeof(*hist) * nbits);
    for (int i = 0; i < d; i++) {
        const int* ai = accu.data() + i * 256;
        int* hi = hist + i * 8;
        for (int j = 0; j < 256; j++)
            for (int k = 0; k < 8; k++)
                if ((j >> k) & 1)
                    hi[k] += ai[j];
    }
}

size_t ivec_checksum(size_t n, const int* a) {
    size_t cs = 112909;
    while (n--)
        cs = cs * 65713 + a[n] * 1686049;
    return cs;
}

namespace {
struct ArgsortComparator {
    const float* vals;
    bool operator()(const size_t a, const size_t b) const {
        return vals[a] < vals[b];
    }
};

struct SegmentS {
    size_t i0; // begin pointer in the permutation array
    size_t i1; // end
    size_t len() const {
        return i1 - i0;
    }
};

// see https://en.wikipedia.org/wiki/Merge_algorithm#Parallel_merge
// extended to > 1 merge thread

// merges 2 ranges that should be consecutive on the source into
// the union of the two on the destination
template <typename T>
void parallel_merge(
        const T* src,
        T* dst,
        SegmentS& s1,
        SegmentS& s2,
        int nt,
        const ArgsortComparator& comp) {
    if (s2.len() > s1.len()) { // make sure that s1 larger than s2
        std::swap(s1, s2);
    }

    // compute sub-ranges for each thread
    std::vector<SegmentS> s1s(nt), s2s(nt), sws(nt);
    s2s[0].i0 = s2.i0;
    s2s[nt - 1].i1 = s2.i1;

    // not sure parallel actually helps here
#pragma omp parallel for num_threads(nt) // 2
    for (int t = 0; t < nt; t++) {
        s1s[t].i0 = s1.i0 + s1.len() * t / nt;
        s1s[t].i1 = s1.i0 + s1.len() * (t + 1) / nt;

        if (t + 1 < nt) { // 求s2s的区间
            T pivot = src
                    [s1s[t].i1]; // s1s下一个区间的起始位置占到了右区间的哪里（偏移）
            size_t i0 = s2.i0, i1 = s2.i1;
            while (i0 + 1 < i1) {
                size_t imed = (i1 + i0) / 2;
                if (comp(pivot, src[imed])) { // pivot < src[imed]
                    i1 = imed;
                } else {
                    i0 = imed;
                }
            }
            s2s[t].i1 = s2s[t + 1].i0 = i1; // 在右区间的偏移量
        }
    }
    s1.i0 = std::min(s1.i0, s2.i0);
    s1.i1 = std::max(s1.i1, s2.i1);
    s2 = s1; // 两段子区间变成同一段！
    sws[0].i0 = s1.i0;
    for (int t = 0; t < nt;
         t++) { // 求每个子线程负责的左右两个子区间归并后的起始位置
        sws[t].i1 = sws[t].i0 + s1s[t].len() + s2s[t].len();
        if (t + 1 < nt) {
            sws[t + 1].i0 = sws[t].i1;
        }
    }
    assert(sws[nt - 1].i1 == s1.i1);

    // do the actual merging
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; t++) {
        SegmentS sw = sws[t];
        SegmentS s1t = s1s[t];
        SegmentS s2t = s2s[t];
        if (s1t.i0 < s1t.i1 && s2t.i0 < s2t.i1) {
            for (;;) {
                // assert (sw.len() == s1t.len() + s2t.len());
                if (comp(src[s1t.i0], src[s2t.i0])) {
                    dst[sw.i0++] = src[s1t.i0++];
                    if (s1t.i0 == s1t.i1)
                        break;
                } else {
                    dst[sw.i0++] = src[s2t.i0++];
                    if (s2t.i0 == s2t.i1)
                        break;
                }
            }
        }
        if (s1t.len() > 0) {
            assert(s1t.len() == sw.len());
            memcpy(dst + sw.i0, src + s1t.i0, s1t.len() * sizeof(dst[0]));
        } else if (s2t.len() > 0) {
            assert(s2t.len() == sw.len());
            memcpy(dst + sw.i0, src + s2t.i0, s2t.len() * sizeof(dst[0]));
        }
    }
}

}; // namespace

void fvec_argsort(size_t n, const float* vals, size_t* perm) {
    for (size_t i = 0; i < n; i++)
        perm[i] = i;
    ArgsortComparator comp = {vals};
    std::sort(perm, perm + n, comp);
}

void fvec_argsort_parallel(size_t n, const float* vals, size_t* perm) {
    size_t* perm2 = new size_t[n];
    // 2 result tables, during merging, flip between them
    size_t *permB = perm2, *permA = perm;

    int nt = omp_get_max_threads();
    { // prepare correct permutation so that the result ends in perm
      // at final iteration
        int nseg = nt;
        while (nseg > 1) {
            nseg = (nseg + 1) / 2;
            std::swap(permA, permB);
            // nt= 2 5-8 17-32 才会交换，因为归并次数是奇数次
            // 最后会落到 perm2 上，
            // 所以起始交换一次，保证最后落到 perm 上
        }
    }

#pragma omp parallel
    for (size_t i = 0; i < n; i++)
        permA[i] = i;

    ArgsortComparator comp = {vals};

    std::vector<SegmentS> segs(nt);

    // independent sorts, 最后不可能有小于nt个元素没有参与排序！余数被考虑到了
#pragma omp parallel for           // n= 100,0009
    for (int t = 0; t < nt; t++) { // 10
        size_t i0 = t * n / nt; // 0 10,0000 20,0001 30,0002 40,0003 50,0004 ...
                                // 90,0008 100,0009
        size_t i1 = (t + 1) * n / nt;
        SegmentS seg = {i0, i1};
        std::sort(permA + seg.i0, permA + seg.i1, comp); // 等分，快排
        segs[t] = seg;
    }
    int prev_nested = omp_get_nested();
    // 启用嵌套并行，即并行时又遇到并行区域，每个线程都会启动多个线程执行一遍嵌套并行区域
    omp_set_nested(1);

    int nseg = nt;                                // 10
    while (nseg > 1) {                            // 10 5 3
        int nseg1 = (nseg + 1) / 2;               // 5 3 2
        int sub_nt = nseg % 2 == 0 ? nt : nt - 1; // 10 9 10
        int sub_nseg1 = nseg / 2;                 // 5 2 1
// 并行归并
#pragma omp parallel for num_threads(nseg1) // 5 3 2
        for (int s = 0; s < nseg; s += 2) { // 0-10 0-5 0-3
            if (s + 1 == nseg) { // otherwise isolated segment 4+1=5 2+1=3
                memcpy(permB + segs[s].i0,
                       permA + segs[s].i0,
                       segs[s].len() * sizeof(size_t));
            } else {
                int t0 = s * sub_nt /
                        sub_nseg1; // (02468)*2 (02)*9/2=(0,9) 0*10*1
                int t1 = (s + 1) * sub_nt /
                        sub_nseg1; // (13579)*2 (13)*9/2=(4,13) 1*10*1
                printf("merge %d %d, %d threads\n",
                       s,
                       s + 1,
                       t1 - t0); // 2 4 10
                parallel_merge(
                        permA,
                        permB,
                        segs[s],
                        segs[s + 1],
                        t1 - t0,
                        comp); // 2 4 10
            }
        }
        for (int s = 0; s < nseg; s += 2) // 02468->01234 024->012
            segs[s / 2] = segs[s];
        nseg = nseg1; // 5 3
        std::swap(permA, permB); // 把 B 上部分归并结果赋值到 A, 即 A 永远是答案
    }
    assert(permA == perm);       // 答案 A 是在输入的 perm 上
    omp_set_nested(prev_nested); // 恢复
    delete[] perm2;
}

const float* fvecs_maybe_subsample( // 从 n 里采样 nmax 个向量
        size_t d,
        size_t* n,
        size_t nmax,
        const float* x,
        bool verbose,
        int64_t seed) {
    if (*n <= nmax) // 没超过上限
        return x; // nothing to do

    size_t n2 = nmax;
    if (verbose) {
        printf("  Input training set too big (max size is %zd), sampling "
               "%zd / %zd vectors\n",
               nmax,
               n2,
               *n);
    }
    std::vector<int> subset(*n);
    rand_perm(subset.data(), *n, seed); // 编号、乱序
    float* x_subset = new float[n2 * d]; // 存储采样结果
    for (int64_t i = 0; i < n2; i++)
        memcpy(&x_subset[i * d], &x[subset[i] * size_t(d)], sizeof(x[0]) * d);
    *n = n2; // 更新训练集个数
    return x_subset;
}

void binary_to_real(size_t d, const uint8_t* x_in, float* x_out) { // 把 d 个二进制位转换为 d 个浮点数，0 转换为 -1，1 还是 1 
    for (size_t i = 0; i < d; ++i) {
        x_out[i] = 2 * ((x_in[i >> 3] >> (i & 7)) & 1) - 1;
    }
}

void real_to_binary(size_t d, const float* x_in, uint8_t* x_out) { // 把 d 个浮点数压缩为 d 位，大于 0 的数所在位编码为 1
    for (size_t i = 0; i < d / 8; ++i) {
        uint8_t b = 0;
        for (int j = 0; j < 8; ++j) {
            if (x_in[8 * i + j] > 0) {
                b |= (1 << j);
            }
        }
        x_out[i] = b;
    }
}

// from Python's stringobject.c
uint64_t hash_bytes(const uint8_t* bytes, int64_t n) { // 求 n 个字节的 hash 值，可用于去重
    const uint8_t* p = bytes;
    uint64_t x = (uint64_t)(*p) << 7;
    int64_t len = n;
    while (--len >= 0) {
        x = (1000003 * x) ^ *p++;
    }
    x ^= n;
    return x;
}

bool check_openmp() {
    omp_set_num_threads(10);

    if (omp_get_max_threads() != 10) {
        return false;
    }

    std::vector<int> nt_per_thread(10);
    size_t sum = 0;
    bool in_parallel = true;
#pragma omp parallel reduction(+ : sum)
    {
        if (!omp_in_parallel()) {
            in_parallel = false;
        }

        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        nt_per_thread[rank] = nt;
#pragma omp for
        for (int i = 0; i < 1000 * 1000 * 10; i++) {
            sum += i;
        }
    }

    if (!in_parallel) {
        return false;
    }
    if (nt_per_thread[0] != 10) {
        return false;
    }
    if (sum == 0) {
        return false;
    }

    return true;
}

} // namespace faiss
