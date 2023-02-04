#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <faiss/IndexFlat.h>

std::mt19937 mt;

double elapsed() { // return in second
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void printSystemTime() {
    std::time_t t = std::time(nullptr);
    std::cout << "\nSystem Time: "
              << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "\n"
              << std::endl;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
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
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 ||
           !"weird file size");    // 字节数是 4d+4 的倍数
    size_t n = sz / ((d + 1) * 4); // 向量个数

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f); // 返回读取float的个数
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

void readDataset(
        const char* trainFileName,
        const char* databaseFileName,
        const char* queryFileName,
        const char* truthFileName,
        size_t& d,
        size_t& nt,
        size_t& nb,
        size_t& nq,
        size_t& k,
        size_t& ng,
        float*& xt,
        float*& xb,
        float*& xq,
        int*& gt) {
    printf("\nBeginReadDatasetFile...\n\n");

    double t0 = elapsed();
    xt = fvecs_read(trainFileName, &d, &nt);
    printf("TrainDataset: d=%ld, nt=%ld, %.1fs\n\n", d, nt, (elapsed() - t0));

    t0 = elapsed();
    size_t db, dq;
    xb = fvecs_read(databaseFileName, &db, &nb);
    printf("DatabaseDataset: d=%ld, nb=%ld, %.1fs\n\n",
           db,
           nb,
           (elapsed() - t0));
    assert(d == db ||
           !"Database vectors does not have same dimension as train set");

    t0 = elapsed();
    xq = fvecs_read(queryFileName, &dq, &nq);
    printf("QueryDataset: d=%ld, nq=%ld, %.1fs\n\n", dq, nq, (elapsed() - t0));
    assert(d == dq ||
           !"Query vectors does not have same dimension as train set");

    t0 = elapsed();
    gt = (int*)fvecs_read(truthFileName, &k, &ng);
    printf("GroundTruthDataset: k=%ld, ng=%ld, %.1fs\n\n",
           k,
           ng,
           (elapsed() - t0));
    assert(ng == nq || !"incorrect number of ground truth vectors");
}

template <typename T>
void writeToFile(int d, int n, const char* filename, T* data) {
    std::fstream aFile;
    aFile.open(filename, std::ios::out); // write,清空再写入
    if (aFile.is_open()) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d - 1; j++) {
                aFile << std::to_string(*data++) << ",";
            }
            aFile << std::to_string(*data++) << "\n";
        }
        aFile.close();
    } else {
        std::cout << "Open " << filename << " fails.\n";
    }
}

void writeDataset( // transform four sift* format file to txt file
        const char* trainFileName,
        const char* databaseFileName,
        const char* queryFileName,
        const char* truthFileName,
        size_t& d,
        size_t& nt,
        size_t& nb,
        size_t& nq,
        size_t& k,
        size_t& ng,
        float*& xt,
        float*& xb,
        float*& xq,
        int*& gt) {
    printf("\nBeginWriteToTxtFile...\n\n");

    double t0 = elapsed();
    if (xt) {
        writeToFile<float>(d, nt, trainFileName, xt);
    }
    printf("TrainDataset: d=%ld, nt=%ld, %.1fs\n\n", d, nt, (elapsed() - t0));

    t0 = elapsed();
    if (xb) {
        writeToFile<float>(d, nb, databaseFileName, xb);
    }
    printf("DatabaseDataset: d=%ld, nb=%ld, %.1fs\n\n",
           d,
           nb,
           (elapsed() - t0));

    t0 = elapsed();
    if (xq) {
        writeToFile<float>(d, nq, queryFileName, xq);
    }
    printf("QueryDataset: d=%ld, nq=%ld, %.1fs\n\n", d, nq, (elapsed() - t0));

    t0 = elapsed();
    if (gt) {
        writeToFile<int>(k, ng, truthFileName, gt);
    }
    printf("GroundTruthDataset: k=%ld, ng=%ld, %.1fs\n\n",
           k,
           ng,
           (elapsed() - t0));
}

void writeSindexInputDoc( // generate doc input dataset of Sindex
        const char* docFileName,
        size_t& d,
        size_t& nb,
        const float* xb) {
    printf("\nBeginWriteToInputDoc\n\n");

    double t0 = elapsed();

    std::vector<size_t> Keys(nb);
    std::iota(Keys.begin(), Keys.end(), 0LL);
    // for (int i = 0; i < nb; i++) {
    //     size_t c = (int)(mt() & 0x7fffffff) % nb;
    //     std::swap(Keys[i], Keys[c]);
    // }

    std::vector<std::string> product_types = {
            "computer",
            "comb",
            "glass",
            "umbrella",
            "bag",
            "handbag",
            "purse",
            "apple",
            "watch",
            "chair",
            "table",
            "match",
            "wallet",
            "door knob",
            "monitor",
            "headset",
            "speaker",
            "power bank",
            "charger",
            "tripod holder",
            "toy",
            "mouse",
            "eye cream",
            "oil",
            "handheld vacuum",
            "Kindle E readers",
            "book",
            "juice",
            "milk",
            "tea",
            "apple",
            "coffee",
            "ice cream"};

    std::fstream aFileStream;
    aFileStream.open(docFileName, std::ios::out); // write,清空再写入
    if (!aFileStream.is_open()) {
        std::cout << "Open " << docFileName << " fails.\n";
        return;
    }

    for (size_t i = 0; i < nb; i++) {
        aFileStream << "{ \"Key\": \"" << Keys[i] << "\", ";
        aFileStream << "\"Fields\": {\"vector_field\": \"";

        std::string vector_float = "";
        for (int j = 0; j < d - 1; j++) {
            vector_float.append(std::to_string(*xb++));
            vector_float.append(",");
        }
        vector_float.append(std::to_string(*xb++));

        aFileStream << vector_float << "\", \"vector_str\": \"" << vector_float
                    << "\", ";
        aFileStream
                << "\"product_type\": \""
                << product_types
                           [(int)(mt() & 0x7fffffff) % product_types.size()];
        aFileStream << "\"}} \n";
        if (i && i % 100000 == 0) {
            printf("generate %d input doc", i);
            printSystemTime();
        }
    }

    aFileStream.close();

    printf("Input doc file [%s] is generated in %.1fs\n\n",
           docFileName,
           (elapsed() - t0));
}

void process_csv( // transform from csv vector file to database and query vector
                  // file in txt format
        const char* oldFilePath,
        const char* databaseFilePath,
        int nquery,
        const char* queryFilePath) {
    std::ifstream oldFileStream;
    oldFileStream.open(oldFilePath, std::ios::in);
    if (!oldFileStream.is_open()) {
        std::cout << "Open " << oldFilePath << " fails.\n";
        return;
    }

    std::fstream dbFileStream;
    dbFileStream.open(databaseFilePath, std::ios::out); // write,清空再写入
    if (!dbFileStream.is_open()) {
        std::cout << "Open " << databaseFilePath << " fails.\n";
        return;
    }

    std::fstream queryFileStream;
    queryFileStream.open(queryFilePath, std::ios::out);
    if (!queryFileStream.is_open()) {
        std::cout << "Open " << queryFilePath << " fails.\n";
        return;
    }

    std::string arow;
    std::vector<std::string> vector_str;
    while (getline(oldFileStream, arow)) {
        int i = arow.find('[');
        int j = arow.find(']');
        arow = arow.substr(i + 1, j - i - 1);
        // std::cout<<arow<<"\n";
        dbFileStream << arow << "\n";
        vector_str.emplace_back(arow);
    }

    int cnt = 0;
    int nb = vector_str.size();
    while (cnt < nquery) {
        queryFileStream << vector_str[(int)(mt() & 0x7fffffff) % nb] << "\n";
        cnt++;
    }

    oldFileStream.close();
    dbFileStream.close();
    queryFileStream.close();
}

float* fvecs_read_txt(
        const char* fname,
        int64_t* d_out,
        int64_t* n_out) { // read dataset in txt file
    int d = -1, d2;
    std::vector<std::vector<float>> data;
    std::ifstream afile(fname);
    if (!afile) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    while (afile) {
        std::string s;
        if (!getline(afile, s)) {
            break;
        }

        std::istringstream ss(s);
        std::vector<float> avector;

        while (ss) {
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

    if (!afile.eof()) {
        std::cerr << "Error!\n";
    }
    afile.close();

    *d_out = d;
    *n_out = data.size();

    float* x = new float[d * data.size()];
    for (int i = 0; i < data.size(); i++) {
        memmove(x + i * d, data[i].data(), d * sizeof(data[i][0]));
    }

    for (int i = 0 * d; i < 1 * d; i++)
        printf("%lf, ", x[i]);
    printf("\n\n");

    return x;
}

void sampleTrainQueryGtDatasets( // random sample query dataset from database
                                 // vector (with noise)
        const char* baseFilePath,
        int64_t ntrain,
        const char* trainFilePath,
        int64_t nquery,
        const char* queryFilePath,
        int64_t k, // 100
        const char* groundTruthPath) {
    int64_t d, nb;
    double t0 = elapsed();
    float* xb = fvecs_read_txt(baseFilePath, &d, &nb);

    printf("read dataset: nb= %lld, d= %lld, [%.2fs].\n",
           nb,
           d,
           elapsed() - t0);
    printSystemTime();

//    t0 = elapsed();
//    std::fstream trainFileStream;
//    trainFileStream.open(trainFilePath, std::ios::out);
//    if (!trainFileStream.is_open()) {
//        std::cout << "Open " << trainFilePath << " fails.\n";
//        return;
//    }
//
//    int cnt = 0;
//    while (cnt < ntrain) { // 可以有重复
//        std::string row = "";
//        int64_t id = (mt() & 0x7fffffff) % (int)nb;
//        int64_t i = id * d;
//        while (i < id * d + d - 1) {
//            row.append(std::to_string(xb[i])).append(",");
//            i++;
//        }
//        row.append(std::to_string(xb[i])).append("\n");
//        trainFileStream << row;
//        cnt++;
//    }
//    trainFileStream.close();
//    printf("Train dataset is generated [%.2fs].\n", elapsed() - t0);
//    printSystemTime();
//
//    t0 = elapsed();
//    float* xq = new float[nquery * d];
//    std::fstream queryFileStream;
//    queryFileStream.open(queryFilePath, std::ios::out);
//    if (!queryFileStream.is_open()) {
//        std::cout << "Open " << queryFilePath << " fails.\n";
//        return;
//    }
//
//    float maxv = *std::max_element(xb, xb + nb * d);
//    float minv = *std::min_element(xb, xb + nb * d);
//    float range = maxv - minv;
//
//    cnt = 0;
//    while (cnt < nquery) {
//        std::string row = "";
//        int64_t id = (mt() & 0x7fffffff) % (int)nb;
//        int64_t qi = cnt * d, bi = id * d;
//        while (qi < cnt * d + d - 1) {
//            // generate query randomly
//            // xq[qi] = minv + float(mt()) / float(mt.max()) * range;
//            // if ((mt() & 1) == 0)
//            // xq[qi] = -xq[qi];
//
//            // generate noise query in half dimension
//            xq[qi] = xb[bi];
//            if ((mt() & 1) == 0) {
//                if ((mt() & 1) == 0)
//                    xq[qi] = xq[qi] + float(mt()) / float(mt.max());
//                else
//                    xq[qi] = xq[qi] - float(mt()) / float(mt.max());
//            }
//
//            row.append(std::to_string(xq[qi])).append(",");
//            qi++;
//            bi++;
//        }
//        // xq[qi] = minv + float(mt()) / float(mt.max()) * range;
//        // if ((mt() & 1) == 0)
//        //     xq[qi] = -xq[qi];
//
//        xq[qi] = xb[bi];
//        if ((mt() & 1) == 0) {
//            if ((mt() & 1) == 0)
//                xq[qi] = xq[qi] + float(mt()) / float(mt.max());
//            else
//                xq[qi] = xq[qi] - float(mt()) / float(mt.max());
//        }
//        row.append(std::to_string(xq[qi])).append("\n");
//        queryFileStream << row;
//        cnt++;
//    }
//    queryFileStream.close();
//    printf("Query dataset is generated [%.2fs].\n", elapsed() - t0);
//    printSystemTime();

// 已有查询向量文件，直接读，生成答案文件
	int64_t nq;
	float* xq = fvecs_read_txt(queryFilePath, &d, &nq);

    t0 = elapsed();
    faiss::IndexFlatL2 index(d);
    printf("FlatL2 is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb); // add vectors to the index
    printf("FlatL2 AvgAddTime= %.3f us, ntotal = %zd\n\n",
           (elapsed() - t0) * 1000000.0 / nb,
           index.ntotal);
    printSystemTime();
    delete[] xb;

    int64_t n_seg = 100; // 每次搜索个数
    int64_t* gt = new int64_t[n_seg * k];
    float* D_truth = new float[n_seg * k];
    for (int64_t i = 0; i < nquery; i += n_seg) {
        t0 = elapsed();
        int64_t n_search = n_seg < nquery - i ? n_seg : nquery - i;
        index.search(n_search, xq + i * d, k, D_truth, gt);

        std::fstream groundTruthFileStream;
        groundTruthFileStream.open(groundTruthPath, std::ios::app);
        if (!groundTruthFileStream.is_open()) {
            std::cout << "Open " << groundTruthPath << " fails.\n";
            return;
        }

        for (int64_t j = 0; j < n_search; j++) {
            std::string row_result = "";
            for (int64_t u = 0; u < k - 1; u++) {
                row_result.append(std::to_string(gt[j * k + u])).append(",");
            }
            row_result.append(std::to_string(gt[j * k + k - 1])).append("\n");
            groundTruthFileStream << row_result;
        }
        groundTruthFileStream.close();
        memset(gt, -1, n_seg * k);
        printf("%d~%d vector ground truth data is stored [%.2fs].\n",
               i,
               i + n_search,
               elapsed() - t0);
        printSystemTime();
    }

    delete[] xq, gt, D_truth;
}

int main() {
    mt = std::mt19937(12345);

    float* xb = nullptr;
    float* xq = nullptr;
    float* xt = nullptr;
    int* gt = nullptr;

    size_t k, d, nt, nb, nq, ng;

    // readDataset(
    //         "sift10K/siftsmall_learn.fvecs",
    //         "sift10K/siftsmall_base.fvecs",
    //         "sift10K/siftsmall_query.fvecs",
    //         "sift10K/siftsmall_groundtruth.ivecs",
    //         d,
    //         nt,
    //         nb,
    //         nq,
    //         k,
    //         ng,
    //         xt,
    //         xb,
    //         xq,
    //         gt);

    // writeDataset(
    //         "sift10K_txt/siftsmall_learn.txt",
    //         "sift10K_txt/siftsmall_base.txt",
    //         "sift10K_txt/siftsmall_query.txt",
    //         "sift10K_txt/siftsmall_groundtruth.txt",
    //         d,
    //         nt,
    //         nb,
    //         nq,
    //         k,
    //         ng,
    //         xt,
    //         xb,
    //         xq,
    //         gt);

    // readDataset(
    //         "sift1M/sift_learn.fvecs",
    //         "sift1M/sift_base.fvecs",
    //         "sift1M/sift_query.fvecs",
    //         "sift1M/sift_groundtruth.ivecs",
    //         d,
    //         nt,
    //         nb,
    //         nq,
    //         k,
    //         ng,
    //         xt,
    //         xb,
    //         xq,
    //         gt);

    // writeDataset(
    //         "sift1M_txt/sift_learn.txt",
    //         "sift1M_txt/sift_base.txt",
    //         "sift1M_txt/sift_query.txt",
    //         "sift1M_txt/sift_groundtruth.txt",
    //         d,
    //         nt,
    //         nb,
    //         nq,
    //         k,
    //         ng,
    //         xt,
    //         xb,
    //         xq,
    //         gt);

    // int64_t d64, nb64;
    // xb = fvecs_read_txt(
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/txt/ID_embedding_data_full_base.txt",
    //         &d64,
    //         &nb64);
    // d = d64, nb = nb64;
    // printSystemTime();
    // writeSindexInputDoc(
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/sindex_input/ID_embedding_data_full_base.input",
    //         d,
    //         nb,
    //         xb);

    // process_csv(
    //         "./ShopeeData/SG_spark_sql_embeddings.csv",
    //         "./ShopeeData/SG_spark_sql_embeddings_base.txt",
    //         10000,
    //         "./ShopeeData/SG_spark_sql_embeddings_query.txt");

    // sampleTrainQueryGtDatasets(
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/txt/ID_embedding_data_full_base.txt",
    //         508679,
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/txt/ID_embedding_data_full_train.txt",
    //         10000,
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/txt/ID_embedding_data_full_noise_query_k100.txt",
    //         100,
    //         "/home/shiwanghua/faiss-main-shopee/demos/ShopeeData/Indonesia/txt/ID_embedding_data_full_noise_groundtruth_k100.txt");

    sampleTrainQueryGtDatasets(
            "/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_base.txt",
            0,
            "/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_learn.txt",
            0,
            "/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_query.txt",
            100,
            "/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_groundtruth.txt");

	sampleTrainQueryGtDatasets(
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_base.txt",
		0,
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_learn.txt",
		0,
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_query.txt",
		100,
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_groundtruth.txt");

    delete[] xb, xq, xt, gt;
    return 0;
}
