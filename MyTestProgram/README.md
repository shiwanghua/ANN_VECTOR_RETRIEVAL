

export MKLROOT=/opt/intel/oneapi/mkl
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -Bbuild -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_GPU=OFF
cmake --build build -v -j6

cd ../; mkdir MyTestProgram; cd MyTestProgram;
mkdir include include/faiss include/faiss/impl include/faiss/utils include/faiss/invlists
cp ../faiss/build/faiss/libfaiss.so ./
cp -r ../faiss/cmake ./

cd ../faiss/faiss
cp AutoTune.h Clustering.h index_factory.h index_io.h Index.h \
   IndexBinary.h IndexFlat.h IndexFlatCodes.h IndexHNSW.h IndexIVF.h \
   IndexPQ.h IndexScalarQuantizer.h MetricType.h \
   ../../MyTestProgram/include/faiss/

cd impl
cp AuxIndexStructures.h DistanceComputer.h FaissAssert.h FaissException.h \
   HNSW.h io.h platform_macros.h PolysemousTraining.h ProductQuantizer-inl.h \
   ProductQuantizer.h Quantizer.h ScalarQuantizer.h \
   ../../../MyTestProgram/include/faiss/impl/

cd ../invlists/
cp DirectMap.h InvertedLists.h ../../../MyTestProgram/include/faiss/invlists/

cd ../utils
cp Heap.h ordered_key_value.h random.h utils.h ../../../MyTestProgram/include/faiss/utils/

cd ../../../MyTestProgram
touch test_libfaiss.cpp CMakeLists.txt


