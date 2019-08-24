#include <CudaUtils.h>
#include <CudaUtils.cuh>
#include <CudaUtils.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

using namespace CudaUtils;

void CudaUtils::cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
		exit(1);
	}
}

void CudaUtils::cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
		exit(1);
	}
}

void *CudaUtils::deviceAllocate(size_t x)
{
	void *p;
	cudaErrCheck(cudaMalloc(&p, x));
	return p;
}

void CudaUtils::deviceFree(void *x)
{
	if (x)
	{
		cudaErrCheck(cudaFree(x));
	}
}

void CudaUtils::memcpyDevice(void *dst, void *src, int len)
{
	cudaErrCheck(cudaMemcpy(dst, src, len, cudaMemcpyDeviceToDevice));
}

void CudaUtils::memcpyDeviceToHost(void *dst, void *src, int len)
{
	cudaErrCheck(cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost));
}

void CudaUtils::memcpyHostToDevice(void *dst, void *src, int len)
{
	cudaErrCheck(cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice));
}

CuBlasHandle::CuBlasHandle() : handle(0)
{ }

CuBlasHandle::~CuBlasHandle()
{
	if (!handle)
		return;
	cublasHandle_t *cuh = static_cast<cublasHandle_t *>(handle);
	cublasErrCheck(cublasDestroy(*cuh));
	delete cuh;
}

void *CuBlasHandle::getHandle()
{
	if (!handle)
	{
		cublasHandle_t *cuh = new cublasHandle_t;
		cublasErrCheck(cublasCreate(cuh));
		handle = cuh;
	}

	return handle;
}

void gemm(MatrixF &lhs, MatrixF &rhs, MatrixF &r, CuBlasHandle &handle)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasHandle_t *cuh = static_cast<cublasHandle_t *>(handle.getHandle());
	cublasErrCheck(cublasSgemm(*cuh,
		CUBLAS_OP_N, CUBLAS_OP_N,
		rhs.cols, lhs.rows, lhs.cols,
		&alpha,
		rhs.data, rhs.cols,
		lhs.data, lhs.cols,
		&beta,
		r.data, r.cols));
}
