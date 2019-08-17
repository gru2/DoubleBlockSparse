#include "utils.h"

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
		exit(1);
	}
}

void *CudaUtils::deviceAllocate(size_t x)
{
	void *p;
	cudaErrCheck(cudaMalloc(&p, x);
	return p;
}

void CudaUtils::deviceFree(void *x)
{
	if (x)
		cudaErrCheck(cudaFree(x));
}
