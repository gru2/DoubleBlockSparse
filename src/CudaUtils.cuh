#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#define cudaErrCheck(stat) { CudaUtils::cudaErrCheck_((stat), __FILE__, __LINE__); }

namespace CudaUtils
{

void cudaErrCheck_(cudaError_t stat, const char *file, int line);

};

#endif
