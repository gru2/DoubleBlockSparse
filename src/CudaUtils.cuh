#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#define cudaErrCheck(stat) { CudaUtils::cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) { CudaUtils::cublasErrCheck_((stat), __FILE__, __LINE__); }

namespace CudaUtils
{

void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line);

};

#endif
