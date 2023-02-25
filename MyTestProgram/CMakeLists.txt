cmake_minimum_required (VERSION 3.16)

project(my_test_program LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FIND_LIBRARY(LIB ./)
set(THIRDPARTY_LIBRARIES ${LIB})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# include_directories(./)
include_directories(./include)

LINK_DIRECTORIES(./)

ADD_EXECUTABLE(test_lib_factory test_lib_factory.cpp)
#ADD_EXECUTABLE(test_speed_all test_speed_all.cpp)
ADD_EXECUTABLE(trans_format trans_format.cpp)

#ADD_EXECUTABLE(test_merge_index test_merge_index.cpp)
#ADD_EXECUTABLE(train_model train_model.cpp)
#ADD_EXECUTABLE(test_io test_io.cpp)
#ADD_EXECUTABLE(test_slice_merge test_slice_merge.cpp)

find_package(OpenMP REQUIRED)
#target_link_libraries(test_slice_merge OpenMP::OpenMP_CXX)

find_package(MKL)
if(MKL_FOUND)
  message(WARNING "MKL Found, come to here")

  target_link_libraries(test_lib_factory ${MKL_LIBRARIES} faiss)
#  target_link_libraries(test_speed_all ${MKL_LIBRARIES} faiss)
  target_link_libraries(trans_format ${MKL_LIBRARIES} faiss)

#  target_link_libraries(train_model ${MKL_LIBRARIES} faiss)
#  target_link_libraries(test_merge_index ${MKL_LIBRARIES} faiss)
#  target_link_libraries(test_io ${MKL_LIBRARIES} faiss)
#  target_link_libraries(test_slice_merge ${MKL_LIBRARIES} faiss)
else()
  message(WARNING "MKL NOT Found, come to here")
#  find_package(BLAS REQUIRED)
  target_link_libraries(test_lib_factory PRIVATE ${BLAS_LIBRARIES} faiss OpenMP::OpenMP_CXX)
#  target_link_libraries(test_speed_all ${BLAS_LIBRARIES} faiss OpenMP::OpenMP_CXX)
  target_link_libraries(trans_format ${BLAS_LIBRARIES} faiss)

#  target_link_libraries(train_model PRIVATE ${BLAS_LIBRARIES} faiss)
#  target_link_libraries(test_merge_index PRIVATE ${BLAS_LIBRARIES} faiss)
#  target_link_libraries(test_io ${MKL_LIBRARIES} faiss)
#  target_link_libraries(test_slice_merge ${MKL_LIBRARIES} faiss)

  #find_package(LAPACK REQUIRED)
  #target_link_libraries(test_lib_factory PRIVATE ${LAPACK_LIBRARIES})
endif()
