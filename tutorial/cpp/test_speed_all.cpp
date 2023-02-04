// 将下面的宏的值设成 1 即可开启对相应索引的测试
// sanity_check 为 true 时进行正确性测试（把前5个数据库向量作为查询向量）
// search_nq 为 true 时把nq个查询向量一次性传进去并行查询，否则每次只传1个查询向量进行串行查询
// IVF有两个参数，对应 TestIVFFlat 里两个for循环

#define TestFlatL2 1
#define TestIVFFlat 1
#define TestPQ 1
#define TestIVFPQ 1
#define TestSQ 1
#define TestHNSWFlat 1
#define TestLSH 1
#define TestNSGFlat 1
#define TestHNSWPQ 1

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
// #include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h> //
#include <faiss/IndexIVFPQ.h> //
#include <faiss/IndexLSH.h> //
#include <faiss/IndexNSG.h> //
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h> //
#include <faiss/index_factory.h>

double elapsed() { // return in second
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

void printSystemTime() {
	std::time_t t = std::time(nullptr);
	// std::cout << "\nSystem Time: " << std::put_time(std::localtime(&t),
	// "%Y-%m-%d %H:%M:%S") << "\n\n";
	std::stringstream ss;
	ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
	printf("\nSystem Time: %s\n\n", ss.str().c_str());
	fflush(stdout);
}

double recallRate(int64_t* I_truth, int64_t* I, int nq, int64_t k) { // 召回率
	double r = 0;
	for (int i = 0; i < nq; ++i) {
		for (int n = 0; n < k; n++) {
			for (int m = 0; m < k; m++) {
				if (I_truth[i * k + n] == I[i * k + m]) {
					r += 1;
					break;
				}
			}
		}
	}
	r = r / (double)(k * nq);
	return r;
}

void printIntervalResults(int from, int to, int64_t* I, float* D, int64_t k) {
	for (int i = from; i < to; i++) {
		for (int j = 0; j < k; j++)
			printf("%5zd ", I[i * k + j]);
		printf("\t[");
		for (int j = 0; j < k; j++)
			printf("%.3f ", D[i * k + j]); // %7g
		printf("]\n");
	}
	printf("\n");
}

// [0,1] 随机分布, 第一维递增
void generateRandomData(
	int64_t d,
	int64_t nb,
	int64_t nq,
	float*& xb,
	float*& xq) {
	xb = new float[d * nb];
	xq = new float[d * nq];
	// xt = new float[d * nt];

	std::mt19937 rng;
	std::uniform_real_distribution<> distribution(0.0, 1.0);
	for (int i = 0; i < nb; i++) {
		for (int j = 0; j < d; j++)
			xb[d * i + j] = distribution(rng);
		xb[d * i] += i / 1000.;
	}
	for (int i = 0; i < nq; i++) {
		for (int j = 0; j < d; j++)
			xq[d * i + j] = distribution(rng);
		xq[d * i] += i / 1000.;
	}

	// for (int i = 0; i < nb; i++) {
	//     for (int j = 0; j < d; j++)
	//         xt[d * i + j] = distrib(rng);
	//     xt[d * i] += i / 1000.;
	// }
}

// 随机正态分布
void generateRandomNormalData(
	double max_mean,
	double max_stddev,
	int64_t d,
	int64_t nb,
	int64_t nq,
	float*& xb,
	float*& xq) {
	std::mt19937 rng;
	std::uniform_real_distribution<> dt_mean(-max_mean, max_mean);
	std::uniform_real_distribution<> dt_dev2(0, max_stddev);

	xb = new float[d * nb];
	xq = new float[d * nq];
	// xt = new float[d * nt];

	for (int i = 0; i < d; i++) {
		double mean = dt_mean(rng);
		double stddev = dt_dev2(rng);
		unsigned seed =
			std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(mean, stddev);

		for (int j = 0; j < nb; j++) {
			xb[j * d + i] = distribution(generator);
		}
		for (int j = 0; j < nq; j++) {
			xq[j * d + i] = distribution(generator);
		}
		// for (int j = 0; j < nq; j++) {
		// xt[j * d + i] = distribution(generator);
		// }
	}
}

// read from a .fvecs / .ivecs file (float/integer)
float* fvecs_read(const char* fname, int64_t* d_out, int64_t* n_out) {
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
	int64_t sz = st.st_size;
	assert(sz % ((d + 1) * 4) == 0 ||
		   !"weird file size");     // 字节数是 4d+4 的倍数
	int64_t n = sz / ((d + 1) * 4); // 向量个数

	*d_out = d;
	*n_out = n;
	float* x = new float[n * (d + 1)];
	int64_t nr = fread(x, sizeof(float), n * (d + 1), f); // 返回读取float的个数
	assert(nr == n * (d + 1) || !"could not read whole file");

	// shift array to remove row headers
	for (int64_t i = 0; i < n; i++)
		memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

	fclose(f);
	return x;
}

void readDatasets_vecs(
	const char* trainFileName,
	const char* databaseFileName,
	const char* queryFileName,
	const char* truthFileName,
	int64_t& d,
	int64_t& nt,
	int64_t& nb,
	int64_t& nq,
	int64_t& k,
	int64_t& ng,
	float*& xt,
	float*& xb,
	float*& xq,
	int64_t*& gt) {
	printf("\nBeginReadDatasetFiles...\n\n");

	double t0 = elapsed();
	xt = fvecs_read(trainFileName, &d, &nt);
	printf("TrainDataset: d=%ld, nt=%ld, %.1fs\n\n", d, nt, (elapsed() - t0));

	t0 = elapsed();
	int64_t db, dq;
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
	int* gt32 = (int*)fvecs_read(truthFileName, &k, &ng);
	gt = new int64_t[ng * k];
	for (int i = 0; i < ng * k; i++)
		gt[i] = (int64_t)gt32[i];
	delete[] gt32;
	printf("GroundTruthDataset: k=%ld, ng=%ld, %.1fs\n\n",
		k,
		ng,
		(elapsed() - t0));
	assert(ng == nq || !"incorrect number of ground truth vectors");
}

float* fvecs_read_txt(const char* fname, int64_t* d_out, int64_t* n_out) {
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
		if (d == -1) {
			d = d2 = avector.size();
		} else
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

void readDatasets_txt(
	const char* trainFileName,
	const char* databaseFileName,
	const char* queryFileName,
	const char* truthFileName,
	int64_t& d,
	int64_t& nt,
	int64_t& nb,
	int64_t& nq,
	int64_t& k,
	int64_t& ng,
	float*& xt,
	float*& xb,
	float*& xq,
	int64_t*& gt) {
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
	float* gt32 = fvecs_read_txt(truthFileName, &k, &ng);
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

int main() {
	printSystemTime();

	int64_t d = 64;      // dimension
	int64_t nb = 100000; // database size
	int64_t nq = 10000;  // nb of queries
	int64_t nt = 50000;  // 训练样本

	int max_mean = 1024; // 随机正态分布的参数
	int max_stddev = 1024;

	int64_t k = 100;

	int nlist = 400; // IVF：聚类个数 ncentroids int(4 * sqrt(nb));
//    int nlist = 10000; // IVF：聚类个数 ncentroids int(4 * sqrt(nb));
	int nprobe = 10;   // IVF：查找时搜索几个类

	int m = 8; // PQ: 把维度分成多少个组，bytes per vector
	int bitsSubcode = 8; // PQ: 把短向量编码成多少个比特的码字，一般都是8

	int m_refine = 2;             // PQR
	int nbits_per_idx_refine = 8; // PQR

	faiss::ScalarQuantizer::QuantizerType qtype_SQ =
		faiss::ScalarQuantizer::QT_8bit_uniform;

	int M_edge_HNSW = 32; // HNSW: 插入点时连接的边数

	int nbits_LSH = 128; // LSH：每个向量编码为多少位

	int R_NSG = 32; // NSG

	bool sanity_check = false; // 为 true 时进行正确性测试（把前5个数据库向量作为查询向量）

	// 为 true时把nq个查询向量一次性传进去，否则每次只传1个查询向量
	bool search_nq = false;

	double t0 = elapsed();

	float* xb = nullptr;
	float* xq = nullptr;
	float* xt = nullptr;
	int64_t* gt = nullptr;

	// generateRandomData(d, nb, nq, xb, xq);
	// generateRandomNormalData(max_mean, max_stddev, d, nb, nq, xb, xq);
	// readDatasets_vecs(
	//         "./sift1M/sift_learn.fvecs",
	//         "./sift1M/sift_base.fvecs",
	//         "./sift1M/sift_query.fvecs",
	//         "./sift1M/sift_groundtruth.ivecs",
	//         d,
	//         nt,
	//         nb,
	//         nq,
	//         k,
	//         nq,
	//         xt,
	//         xb,
	//         xq,
	//         gt);

	// xb = fvecs_read_txt(
	//         ".../_base.txt",
	//         &d,
	//         &nb);
	// xq = fvecs_read_txt(
	//         ".../...query.txt",
	//         &d,
	//         &nq);

	readDatasets_txt(
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_learn.txt",
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_base.txt",
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_query.txt",
		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift10K_txt/siftsmall_groundtruth.txt",
//		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_learn.txt",
//		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_base.txt",
//		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_query.txt",
//		"/home/swh/桌面/MyProgram/Faiss/dataset/siftDataset/sift1M_txt/sift_groundtruth.txt",
		d,
		nt,
		nb,
		nq,
		k,
		nq,
		xt,
		xb,
		xq,
		gt);

	int64_t* I_truth = new int64_t[k * nq]; // 拿Flat的答案作为正确答案
	float* D_truth = new float[k * nq];

	printf("\nd= %d, nb= %d, nt= %d, nq= %d, k= %d, nlist= %d , nprobe=%d, \n\
m= %d, bitsSubcode= %d, m_refine= %d, nbits_per_idx_refine= %d,\n\
M_edge_HNSW= %d, nbits_LSH= %d, R_NSG= %d, sanity_check= %d, search_nq= %d\n\n",
		d,
		nb,
		nt,
		nq,
		k,
		nlist,
		nprobe,
		m,
		bitsSubcode,
		m_refine,
		nbits_per_idx_refine,
		M_edge_HNSW,
		nbits_LSH,
		R_NSG,
		sanity_check,
		search_nq);
	printSystemTime();

	// if (gt) {
	//     printf("The last 5 ground truth:\n");
	//     printIntervalResults(nq - 5, nq, gt, D_truth, k);
	// }

// Test-1 IndexFlatL2
#if TestFlatL2
	{
		faiss::IndexFlatL2 index(d); // call constructor
		printf("FlatL2 is_trained = %s\n", index.is_trained ? "true" : "false");
		double begin = elapsed();
		index.add(nb, xb); // add vectors to the index
		printf("FlatL2 AvgAddTime= %.3f us, ntotal = %zd\n\n",
			(elapsed() - begin) * 1000000.0 / nb,
			index.ntotal);
		printSystemTime();

		if (sanity_check) { // sanity check: search 5 first vectors of xb
			int64_t* I = new int64_t[k * 5];
			float* D = new float[k * 5];

			index.search(5, xb, k, D, I);

			printf("IVF_sanity_check:\nI=\n");
			printIntervalResults(0, 5, I, D, k);

			delete[] I;
			delete[] D;
		}

		{ // search xq
			begin = elapsed();
			if (search_nq) {
				printf("numThreads: %d\n\n", omp_get_max_threads());
				index.search(nq, xq, k, D_truth, I_truth);
			} else {
				for (int i = 0; i < nq; i++) // 避免并行查询
					index.search(
						1, xq + i * d, k, D_truth + i * k, I_truth + i * k);
			}
			printf("FlatL2 AvgSearchTime= %.3f ms\n\n",
				(elapsed() - begin) * 1000.0 / nq);
			printf("Recall@%d: %f\n\n",
				k,
				recallRate(gt ? gt : I_truth, I_truth, nq, k));

			// printf("FlatL2 GroundTruth (1 last results)=\n");
			// printIntervalResults(nq - 1, nq, I_truth, D_truth, k);
		}

		printSystemTime();
	}
#endif

// Test 2-IVFFlat
#if TestIVFFlat
	{
		int nlists[12] = {
			50,
			100,
			200,
			500,
			1000,
			2000,
			5000,
			10000,
			15000,
			20000,
			25000,
			50000};
		int nprobes[25] = {1,   2,   3,   4,   5,   8,   10,  12,
						   15,  20,  25,  30,  40,  50,  100, 150,
						   200, 250, 300, 325, 500, 750, 1000};
		for (int t = 0; t < 4; t++) {
			// nlist = nlists[t];
			nprobe = nprobes[t];
			faiss::IndexFlatL2 quantizer(d); // the other index
			faiss::IndexIVFFlat index(&quantizer, d, nlist);
			assert(!index.is_trained);

			double begin = elapsed();
			if (xt)
				index.train(nt, xt);
			else
				index.train(nb, xb);
			printf("\n\nIVF%dFlatL2, nprobe=%d, TotalTrainTime= %.3f us\n",
				nlist,
				nprobe,
				(elapsed() - begin) * 1000000.0);
			printSystemTime();
			assert(index.is_trained);

			begin = elapsed();
			index.add(nb, xb); // add vectors to the index
			printf("IVF%dFlatL2 AvgAddTime= %.3f us, ntotal = %zd\n\n",
				nlist,
				(elapsed() - begin) * 1000000.0 / nb,
				index.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index.search(5, xb, k, D, I);

				printf("IVF%dFlatL2_sanity_check:\nI=\n", nlist);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				index.nprobe = nprobe;
				begin = elapsed();
				if (search_nq) {
					index.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index.search(1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("IVF%dFlatL2 AvgSearchTime= %.3f ms\n",
					nlist,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				printf("I_IVF%d-%dFlatL2 (5 last results)=\n", nlist, nprobe);
				printIntervalResults(nq - 1, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
		for (int t = 6; t < 6; t++) {
			nlist = nlists[t];
			nprobe = 80 * k * nlist / nb;
			faiss::IndexFlatL2 quantizer(d); // the other index
			faiss::IndexIVFFlat index(&quantizer, d, nlist);
			assert(!index.is_trained);

			double begin = elapsed();
			if (xt)
				index.train(nt, xt);
			else
				index.train(nb, xb);
			printf("\n\nIVF%dFlatL2, nprobe=%d, TotalTrainTime= %.3f us\n",
				nlist,
				nprobe,
				(elapsed() - begin) * 1000000.0);
			printSystemTime();
			assert(index.is_trained);

			begin = elapsed();
			index.add(nb, xb); // add vectors to the index
			printf("IVF%dFlatL2 AvgAddTime= %.3f us, ntotal = %zd\n\n",
				nlist,
				(elapsed() - begin) * 1000000.0 / nb,
				index.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index.search(5, xb, k, D, I);

				printf("IVF%dFlatL2_sanity_check:\nI=\n", nlist);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				index.nprobe = nprobe;
				begin = elapsed();
				if (search_nq) {
					index.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index.search(1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("IVF%dFlatL2 AvgSearchTime= %.3f ms\n",
					nlist,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_IVF%d-%dFlatL2 (5 last results)=\n", nlist,
				// nprobe); printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif
// Test 3-PQ
#if TestPQ
	{
		int ms[3] = {4, 8, 16};
		for (int t = 0; t < 3; t++) {
			m = ms[t];
			faiss::IndexPQ indexpq(d, m, bitsSubcode);
			assert(!indexpq.is_trained);

			double begin = elapsed();
			if (xt)
				indexpq.train(nt, xt);
			else
				indexpq.train(nb, xb);
			printf("\n\nPQ%d-%d, TotalTrainTime= %.3f us\n",
				m,
				bitsSubcode,
				(elapsed() - begin) * 1000000.0);
			assert(indexpq.is_trained);
			printSystemTime();

			begin = elapsed();
			indexpq.add(nb, xb); // add vectors to the index
			printf("PQ%d_%d AvgAddTime= %.3f us, ntotal = %zd\n\n",
				m,
				bitsSubcode,
				(elapsed() - begin) * 1000000.0 / nb,
				indexpq.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				indexpq.search(5, xb, k, D, I);

				printf("PQ%d_%d_sanity_check:\nI=\n", m, bitsSubcode);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				begin = elapsed();
				if (search_nq) {
					indexpq.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						indexpq.search(1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("PQ%d_%d AvgSearchTime= %.3f ms\n",
					m,
					bitsSubcode,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("PQ%d_%d (5 last results)=\n", m, bitsSubcode);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

// Test 4-IVFPQ
#if TestIVFPQ
	{
		int ms[3] = {4, 8, 16};
		for (int i = 0; i < 4; i++) {
			// nlist = 10000;
			m = ms[i];
			faiss::IndexFlatL2 coarse_quantizer(d); // the other index
			faiss::IndexIVFPQ index(&coarse_quantizer, d, nlist, m, 8);

			double begin = elapsed();
			if (xt)
				index.train(nt, xt);
			else
				index.train(nb, xb);
			printf("\n\nIVF%dPQ%d TotalTrainTime= %.3f us\n",
				nlist,
				m,
				(elapsed() - begin) * 1000000.0);
			printSystemTime();

			begin = elapsed();
			index.add(nb, xb);
			printf("IVF%dPQ%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
				nlist,
				m,
				(elapsed() - begin) * 1000000.0 / nb,
				index.ntotal);
			printSystemTime();

			printf("[%.3f s] imbalance factor: %g\n",
				elapsed() - t0,
				index.invlists->imbalance_factor());

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index.search(5, xb, k, D, I);

				printf("IVF%dPQ%d_sanity_check:\nI=\n", nlist, m);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				index.nprobe = nprobe;
				begin = elapsed();
				if (search_nq) {
					index.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index.search(1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("IVF%dPQ%d AvgSearchTime= %.3f ms\n",
					nlist,
					m,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_IVF%dPQ%d (5 last results)=\n", nlist, m);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

// Test 6-SQ
#if TestSQ
	{
		faiss::IndexScalarQuantizer index_sq(d, qtype_SQ);

		double begin = elapsed();
		if (xt)
			index_sq.train(nt, xt);
		else
			index_sq.train(nb, xb);
		printf("\n\nSQ%d TotalTrainTime= %.3f us\n",
			qtype_SQ,
			(elapsed() - begin) * 1000000.0);

		begin = elapsed();
		index_sq.add(nb, xb);
		printf("SQ%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
			qtype_SQ,
			(elapsed() - begin) * 1000000.0 / nb,
			index_sq.ntotal);

		if (sanity_check) { // sanity check
			int64_t* I = new int64_t[k * 5];
			float* D = new float[k * 5];

			index_sq.search(5, xb, k, D, I);

			printf("SQ%d_sanity_check:\nI=\n", qtype_SQ);
			printIntervalResults(0, 5, I, D, k);

			delete[] I;
			delete[] D;
		}

		{ // search xq
			int64_t* I = new int64_t[k * nq];
			float* D = new float[k * nq];

			begin = elapsed();
			if (search_nq) {
				index_sq.search(nq, xq, k, D, I);
			} else {
				for (int i = 0; i < nq; i++) // 避免并行查询
					index_sq.search(1, xq + i * d, k, D + i * k, I + i * k);
			}
			printf("SQ%d AvgSearchTime= %.3f ms\n",
				qtype_SQ,
				(elapsed() - begin) * 1000.0 / nq);
			printf("Recall@%d: %f\n\n",
				k,
				recallRate(gt ? gt : I_truth, I, nq, k));

			printf("I_SQ%d (5 last results)=\n", qtype_SQ);
			printIntervalResults(nq - 1, nq, I, D, k);

			delete[] I;
			delete[] D;
		}
	}
#endif

// Test 7-IndexHNSWFlat
#if TestHNSWFlat
	{
		int medges[6] = {4, 8, 96, 128};
		for (int t = 2; t < 3; t++) {
			M_edge_HNSW = medges[t];
			faiss::IndexHNSWFlat index_hnsw(d, M_edge_HNSW);
			index_hnsw.setEFParameter(100, 100);
			double begin = elapsed();
			if (xt)
				index_hnsw.train(nt, xt);
			else
				index_hnsw.train(nb, xb);
			printf("\n\nHNSW%dFlat TotalTrainTime= %.3f us\n",
				M_edge_HNSW,
				(elapsed() - begin) * 1000000.0);

			begin = elapsed();
			index_hnsw.add(nb, xb);
			printf("HNSWFlat%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
				M_edge_HNSW,
				(elapsed() - begin) * 1000000.0 / nb,
				index_hnsw.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index_hnsw.search(5, xb, k, D, I);

				printf("HNSWFlat%d_sanity_check:\nI=\n", M_edge_HNSW);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				begin = elapsed();
				if (search_nq) {
					index_hnsw.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index_hnsw.search(
							1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("HNSWFlat%d AvgSearchTime= %.3f ms\n",
					M_edge_HNSW,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_HNSWFlat%d (5 last results)=\n", M_edge_HNSW);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

// Test 8-LSH
#if TestLSH
	{
		const int n_LSH_exp = 7;
		int nbitss[n_LSH_exp] = {32, 64, 128, 256, 512, 1024, 2048};
		for (int t = 0; t < 4; t++) {
			nbits_LSH = nbitss[t];
			faiss::IndexLSH index_lsh(d, nbits_LSH);

			double begin = elapsed();
			if (xt)
				index_lsh.train(nt, xt);
			else
				index_lsh.train(nb, xb);
			printf("\n\nLSH%d TotalTrainTime= %.3f us\n",
				nbits_LSH,
				(elapsed() - begin) * 1000000.0);

			begin = elapsed();
			index_lsh.add(nb, xb);
			printf("LSH%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
				nbits_LSH,
				(elapsed() - begin) * 1000000.0 / nb,
				index_lsh.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index_lsh.search(5, xb, k, D, I);

				printf("LSH%d_sanity_check:\nI=\n", nbits_LSH);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				begin = elapsed();
				if (search_nq) {
					index_lsh.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index_lsh.search(
							1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("LSH%d AvgSearchTime= %.3f ms\n",
					nbits_LSH,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_LSH%d (5 last results)=\n", nbits_LSH);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

// Test 9-NSGFlat
#if TestNSGFlat
	{
		int rs[6] = {4, 8, 16, 32, 64, 128};
		for (int t = 6; t < 6; t++) {
			R_NSG = rs[t];
			faiss::IndexNSGFlat index_nsg_flat(d, R_NSG);

			double begin = elapsed();
			if (xt)
				index_nsg_flat.train(nt, xt);
			else
				index_nsg_flat.train(nb, xb);
			printf("\n\nNSG%d TotalTrainTime= %.3f us\n",
				R_NSG,
				(elapsed() - begin) * 1000000.0);

			begin = elapsed();
			index_nsg_flat.add(nb, xb);
			printf("NSG%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
				R_NSG,
				(elapsed() - begin) * 1000000.0 / nb,
				index_nsg_flat.ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index_nsg_flat.search(5, xb, k, D, I);

				printf("NSG%d_sanity_check:\nI=\n", R_NSG);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				begin = elapsed();
				if (search_nq) {
					index_nsg_flat.search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index_nsg_flat.search(
							1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("NSG%d AvgSearchTime= %.3f ms\n",
					R_NSG,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_NSG%d (5 last results)=\n", R_NSG);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

// Test 10-HNSWPQ
#if TestHNSWPQ
	{
		int medges[6] = {4, 8, 64, 128};
		for (int t = 2; t < 4; t++) {
			M_edge_HNSW = medges[t];
			// faiss::IndexHNSWFlat index_hnsw(d, M_edge_HNSW);
			faiss::Index* index_hnsw =
				faiss::index_factory((int)d, "HNSW128_PQ8");

			double begin = elapsed();
			if (xt)
				index_hnsw->train(nt, xt);
			else
				index_hnsw->train(nb, xb);
			printf("\n\nHNSW%dFlat TotalTrainTime= %.3f us\n",
				M_edge_HNSW,
				(elapsed() - begin) * 1000000.0);

			begin = elapsed();
			index_hnsw->add(nb, xb);
			printf("HNSWFlat%d AvgAddTime= %.3f us\nntotal = %zd\n\n",
				M_edge_HNSW,
				(elapsed() - begin) * 1000000.0 / nb,
				index_hnsw->ntotal);
			printSystemTime();

			if (sanity_check) { // sanity check
				int64_t* I = new int64_t[k * 5];
				float* D = new float[k * 5];

				index_hnsw->search(5, xb, k, D, I);

				printf("HNSWFlat%d_sanity_check:\nI=\n", M_edge_HNSW);
				printIntervalResults(0, 5, I, D, k);

				delete[] I;
				delete[] D;
			}

			{ // search xq
				int64_t* I = new int64_t[k * nq];
				float* D = new float[k * nq];

				begin = elapsed();
				if (search_nq) {
					index_hnsw->search(nq, xq, k, D, I);
				} else {
					for (int i = 0; i < nq; i++) // 避免并行查询
						index_hnsw->search(
							1, xq + i * d, k, D + i * k, I + i * k);
				}
				printf("HNSWFlat%d AvgSearchTime= %.3f ms\n",
					M_edge_HNSW,
					(elapsed() - begin) * 1000.0 / nq);
				printf("Recall@%d: %f\n\n",
					k,
					recallRate(gt ? gt : I_truth, I, nq, k));

				// printf("I_HNSWFlat%d (5 last results)=\n", M_edge_HNSW);
				// printIntervalResults(nq - 5, nq, I, D, k);

				delete[] I;
				delete[] D;
			}
			printSystemTime();
		}
	}
#endif

	delete[] xb, xq, xt, gt, I_truth, D_truth;

	return 0;
}
