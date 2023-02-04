#include <faiss/index_factory.h>
#include <faiss/AutoTune.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define autoTuneParameter 1

#define TestFlatL2 1
#define TestIVFFlat 1
#define TestPQ 0
#define TestIVFPQ 0
#define TestIVFPQR 0
#define TestSQ 0
#define TestHNSWFlat 0
#define TestLSH 0
#define TestNSGFlat 0

double elapsed()
{ // return in second
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double recallRate(int64_t *I_truth, int64_t *I, int nq, int64_t k)
{ // 召回率
    double r = 0;
    for (int i = 0; i < nq; ++i)
    {
        for (int n = 0; n < k; n++)
        {
            for (int m = 0; m < k; m++)
            {
                if (I_truth[i * k + n] == I[i * k + m])
                {
                    r += 1;
                    break;
                }
            }
        }
    }
    r = r / (double)(k * nq);
    return r;
}

void printIntervalResults(int from, int to, int64_t *I, float *D, int64_t k)
{
    for (int i = from; i < to; i++)
    {
        for (int j = 0; j < k; j++)
            printf("%5zd ", I[i * k + j]);
        printf("\t[");
        for (int j = 0; j < k; j++)
            printf("%.3f ", D[i * k + j]); // %7g
        printf("]\n");
    }
    printf("\n");
}

float *fvecs_read(const char *fname, int64_t *d_out, int64_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if (!f)
    {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f),
          &st); // fileno：返回数据流的文件句柄，fstat：根据已打开的文件描述符取得文件的状态
    int64_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 ||
           !"weird file size");     // 字节数是 4d+4 的倍数
    int64_t n = sz / ((d + 1) * 4); // 向量个数

    *d_out = d;
    *n_out = n;
    float *x = new float[n * (d + 1)];
    int64_t nr = fread(x, sizeof(float), n * (d + 1), f); // 返回读取float的个数
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (int64_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

float *fvecs_read_txt(const char *fname, int64_t *d_out, int64_t *n_out)
{
    int d = -1, d2;
    std::vector<std::vector<float>> data;
    std::ifstream afile(fname);
    if (!afile)
    {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    while (afile)
    {
        std::string s;
        if (!getline(afile, s))
        {
            break;
        }

        std::istringstream ss(s);
        std::vector<float> avector;

        while (ss)
        {
            std::string fnumber;
            if (!getline(ss, fnumber, ','))
                break;
            avector.push_back(atof(fnumber.c_str()));
        }
        if (d == -1)
            d = d2 = avector.size();
        else
            d2 = avector.size();
        assert((d > 0 && d < 1000000 && d == d2) || !"unreasonable dimension");

        data.push_back(avector);

        // printf("%ld\n",data.size());
    }

    if (!afile.eof())
    {
        std::cerr << "Error!\n";
    }
    afile.close();

    *d_out = d;
    *n_out = data.size();

    float *x = new float[d * data.size()];
    for (int i = 0; i < data.size(); i++)
    {
        memmove(x + i * d, data[i].data(), d * sizeof(data[i][0]));
    }

    for (int i = 0 * d; i < 1 * d; i++)
        printf("%lf, ", x[i]);
    printf("\n\n");

    return x;
}

void readDatasets_txt(
    const char *trainFileName,
    const char *databaseFileName,
    const char *queryFileName,
    const char *truthFileName,
    int64_t &d,
    int64_t &nt,
    int64_t &nb,
    int64_t &nq,
    int64_t &k,
    int64_t &ng,
    float *&xt,
    float *&xb,
    float *&xq,
    int64_t *&gt)
{
    printf("\nBeginReadDatasetFile...\n\n");

    double t0 = elapsed();
    xt = fvecs_read_txt(trainFileName, &d, &nt);
    printf("TrainDataset: d=%ld, nt=%ld, %.1fs\n\n", d, nt, (elapsed() - t0));
    printf("******************************************************************************************\n\n");

    t0 = elapsed();
    int64_t db, dq;
    xb = fvecs_read_txt(databaseFileName, &db, &nb);
    printf("DatabaseDataset: d=%ld, nb=%ld, %.1fs\n\n",
           db,
           nb,
           (elapsed() - t0));
    printf("\n******************************************************************************************\n\n");
    assert(d == db ||
           !"Database vectors does not have same dimension as train set");

    t0 = elapsed();
    xq = fvecs_read_txt(queryFileName, &dq, &nq);
    printf("QueryDataset: d=%ld, nq=%ld, %.1fs\n\n", dq, nq, (elapsed() - t0));
    printf("\n******************************************************************************************\n\n");
    assert(d == dq ||
           !"Query vectors does not have same dimension as train set");

    t0 = elapsed();
    float *gt32 = fvecs_read_txt(truthFileName, &k, &ng);
    gt = new int64_t[ng * k];
    for (int i = 0; i < ng * k; i++)
        gt[i] = (int64_t)gt32[i];
    delete[] gt32;
    printf("GroundTruthDataset: k=%ld, ng=%ld, %.1fs\n\n",
           k,
           ng,
           (elapsed() - t0));
    printf("\n******************************************************************************************\n\n");
    assert(ng == nq || !"incorrect number of ground truth vectors");
}

int main()
{
    int64_t d = 128;     // dimension
    int64_t nb = 100000; // database size
    int64_t nq = 10000;  // nb of queries
    int64_t nt = 50000;  // 训练样本

    int nlist = 4096;
    int nprobe = 10;

    int64_t k = 10;

    // 为 true时把nq个查询向量一次性传进去，否则每次只传1个查询向量
    bool search_nq = false;

    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char* index_key = "IVF64,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    // const char *index_key = "Flat";                      //  100% 0.13~1.27ms -
    // const char *index_key = "PQ8";                       // 67.7% 0.05ms * 50.1% 0.03ms
    // const char *index_key = "PCA80,Flat";                // 93.3% 0.12ms -
    const char *index_key = "IVF200,Flat"; // 67.3% 0.08ms *  100% 0.50ms
    // const char *index_key = "IVF40,PQ8";                 // 41.7% 0.03ms * 67.4% 0.12ms
    // const char *index_key = "IVF40,SQ8";                 // 46.0% 1.00ms * 99.3% 1.11ms
    // const char *index_key = "IVF10,PQ4+8";               // 57.8% 0.26ms * 69.6% 0.21ms  3个参数
    // const char *index_key = "IMI2x4,PQ8";                // 31.6% 0.06ms * 61.9% 0.07ms  3个参数
    // const char *index_key = "IMI2x2,PQ4+8";              // 56.9% 0.82ms * 74.4% 2.01ms  4个参数
    // const char *index_key = "OPQ4_8,IMI2x2,PQ2+4";       // 34.3% 0.17ms * 32.6% 0.05ms 4个参数
    // const char *index_key = "HNSW32,Flat"; // (Flat可省)  // 85.5% 0.07ms * 95.1% 1.00ms
    // const char *index_key = "HNSW32_PQ8";                // 66.0% 0.06ms * 67.8% 0.07ms efSearch
    // const char *index_key = "LSHr";   // LSHr LSHt       // 44.8% 0.06ms -
    // const char *index_key = "NSG16"; // 32: 99.4% 0.26ms // 98.8% 0.11ms -
    // const char *index_key = "NSG32_PQ8";                 // 67.7% 0.05ms -

    float *xb = nullptr;
    float *xq = nullptr;
    float *xt = nullptr;
    int64_t *gt = nullptr;

    // readDatasets_txt(
    //     "./sift10K_txt/siftsmall_learn.txt",
    //     "./sift10K_txt/siftsmall_base.txt",
    //     "./sift10K_txt/siftsmall_query.txt",
    //     "./sift10K_txt/siftsmall_groundtruth.txt",
    //     d,
    //     nt,
    //     nb,
    //     nq,
    //     k,
    //     nq,
    //     xt,
    //     xb,
    //     xq,
    //     gt);
    xb = fvecs_read_txt("/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/SG_spark_sql_embeddings_base.txt", &d, &nb);
    xq = fvecs_read_txt("/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/SG_spark_sql_embeddings_query.txt", &d, &nq);

    printf("nb= %lld, nq= %lld, d= %lld, search_nq= %d\n\n", nb, nq, d, search_nq);

    faiss::Index *index;

    int64_t *I_truth = new int64_t[k * nq]; // 拿Flat的答案作为正确答案
    float *D_truth = new float[k * nq];

// Test-1 IndexFlatL2
#if TestFlatL2
    {
        index = faiss::index_factory((int)d, "Flat");
        printf("FlatL2 is_trained = %s\n", index->is_trained ? "true" : "false");
        double begin = elapsed();
        index->add(nb, xb); // add vectors to the index
        printf("FlatL2 AvgAddTime= %.3f us, ntotal = %zd, d=%lld\n\n", (elapsed() - begin) * 1000000.0 / nb, index->ntotal, d);

        { // search xq
            double begin = elapsed();
            if (search_nq)
            {
                printf("numThreads: %d\n\n", omp_get_max_threads());
                index->search(nq, xq, k, D_truth, I_truth);
            }
            else
            {
                for (int i = 0; i < nq; i++) // 避免并行查询
                    index->search(1, xq + i * d, k, D_truth + i * k, I_truth + i * k);
            }
            printf("FlatL2 AvgSearchTime= %.3f ms\n\n", (elapsed() - begin) * 1000.0 / nq);
            printf("Recall@%d: %f\n\n", k, recallRate(gt ? gt : I_truth, I_truth, nq, k));

            // printf("FlatL2 GroundTruth (5 last results)=\n");
            // printIntervalResults(nq - 5, nq, I_truth, D_truth, k);
        }

        delete index;
        index = nullptr;
    }
#endif

    index = faiss::index_factory((int)d, index_key);

    double begin = elapsed();
    if (xt)
        index->train(nt, xt);
    else
        index->train(nb, xb);
    printf("\n\n%s TotalTrainTime= %.3f us\n",
           index_key,
           (elapsed() - begin) * 1000000.0);
    assert(index->is_trained);

    begin = elapsed();
    index->add(nb, xb); // add vectors to the index
    printf("%s AvgAddTime= %.3f us, ntotal = %zd\n\n",
           index_key,
           (elapsed() - begin) * 1000000.0 / nb,
           index->ntotal);

    // Result of the auto-tuning
#if autoTuneParameter
    { // auto-tuning
        begin = elapsed();
        std::string selected_params;
        faiss::OneRecallAtRCriterion crit(nq, 1);
        crit.set_groundtruth(k, nullptr, gt ? gt : I_truth);
        crit.nnn = k; // by default, the criterion will request only 1 NN

        faiss::ParameterSpace params;
        params.initialize(index);

        printf("Preparing auto-tune parameters (%.3fs)\n\n", elapsed() - begin);
        printf("Auto-tuning over %ld parameters (%ld combinations)\n\n", params.parameter_ranges.size(), params.n_combinations());

        begin = elapsed();

        faiss::OperatingPoints ops;
        params.explore(index, nq, xq, crit, &ops); // 可能这里就修改了index让其更快了

        printf("\nFound the following operating points (%.3fs): \n\n", elapsed() - begin);
        ops.display();

        // get the biggest 1-recall@1
        if (ops.optimal_pts.size() > 0 && ops.optimal_pts[0].perf)
        {
            selected_params = ops.optimal_pts[0].key;
            printf("%d: %.3f\n", 0, ops.optimal_pts[0].perf);
        }
        for (int i = 1, besti = 0; i < ops.optimal_pts.size(); i++)
        {
            printf("%d: %.3f\n", ops.optimal_pts[i].perf);
            if (ops.optimal_pts[i].perf > ops.optimal_pts[besti].perf)
            {
                selected_params = ops.optimal_pts[i].key;
                besti = i;
            }
        }

        // assert(selected_params.size() >=0 || !"could not find good enough op point");

        printf("\nSetting parameter configuration \"%s\" on index\n\n", selected_params.c_str());

        params.set_index_parameters(index, selected_params.c_str());
    }
#endif

    { // search xq

        int64_t *I = new int64_t[k * nq];
        float *D = new float[k * nq];

        double begin = elapsed();
        if (search_nq)
        {
            index->search(nq, xq, k, D, I);
        }
        else
        {
            for (int i = 0; i < nq; i++) // 避免并行查询
                index->search(1, xq + i * d, k, D + i * k, I + i * k);
        }
        printf("%s AvgSearchTime= %.3f ms\n", index_key, (elapsed() - begin) * 1000.0 / nq);
        printf("Recall@%d: %f\n\n", k, recallRate(gt ? gt : I_truth, I, nq, k));

        printf("I_%s (5 last results)=\n", index_key);
        printIntervalResults(nq - 5, nq, I, D, k);

        delete[] I;
        delete[] D;
    }

    printf("DONE.\n");

    delete[] xb, xq, xt, gt, I_truth, D_truth;
    return 0;
}