set(CMAKE_DEBUG_POSTFIX "d")

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES CudaUtils.cpp)

set(HEADER_FILES CudaUtils.h CudaUtils.cuh)

set(TESTS test_double_block_sparse.cpp)

include_directories(. ../ext/GVal ../ext/Usutf ../ext/eigen ../ext/cub)

add_library(double_block_sparse STATIC ${SOURCES} ${HEADER_FILES})
add_executable(test_double_block_sparse ${TESTS} ${HEADER_FILES})
target_link_libraries(test_double_block_sparse double_block_sparse)

CUDA_ADD_CUBLAS_TO_TARGET(test_double_block_sparse)
