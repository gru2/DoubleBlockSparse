set(CMAKE_DEBUG_POSTFIX "d")

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_35")

set(SOURCES CudaUtils.cu EigenUtils.cpp matmul_op_gpu.cu)

set(HEADER_FILES CudaUtils.h CudaUtils.cuh EigenUtils.h)

set(TESTS test_double_block_sparse.cpp)

include_directories(. ../ext/Usutf ../ext/eigen ../ext/cub)
link_directories(../ext/Usutf/build )

add_library(double_block_sparse STATIC ${SOURCES} ${HEADER_FILES})
add_executable(test_double_block_sparse ${TESTS} ${HEADER_FILES})
target_link_libraries(test_double_block_sparse double_block_sparse Usutf)

CUDA_ADD_CUBLAS_TO_TARGET(test_double_block_sparse)
