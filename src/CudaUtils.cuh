#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cublas_v2.h>
#include <cuda.h>

#define cudaErrCheck(stat) { CudaUtils::cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) { CudaUtils::cublasErrCheck_((stat), __FILE__, __LINE__); }

namespace CudaUtils
{

void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line);

};

bool Gemm_TN(CUstream stream, uint SMs,
	float* u,
	const float* xf,
	const float* ef,
	uint C, uint K, uint N);

#endif
