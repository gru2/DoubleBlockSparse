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

void CudaUtils::gemm(MatrixF &r, const MatrixF &lhs, const MatrixF &rhs, CuBlasHandle &handle)
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

<template int TILE_SIZE>
__global__ void gemmTiledKernel(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int N, int M, int K)
{
	int tidx = threadIdx.x + blockIdx.x * TILE_SIZE;
	int tidy = threadIdx.y + blockIdx.y * TILE_SIZE;
	int Tx = threadIdx.x;
	int Ty = threadIdx.y;
	int Bx = blockIdx.x;
	int By = blockIdx.y;

	__shared__ float As[TILE_SIZE * TILE_SIZE];
	__shared__ float Bs[TILE_SIZE * TILE_SIZE];

	float s = 0.0f;
	int n = K / TILE_SIZE;
	for (int kB = 0; kB < n; kB++)
	{
		// Copy tile data from A and B into shared memory.
		As[Ty * TILE_SIZE + Tx] = A[tidy * K + kB * TILE_SIZE + Tx];
		Bs[Tx * TILE_SIZE + Ty] = B[(kB * TILE_SIZE + Ty) * N + tidx];
		__syncthreads();

		// Multiply tiles.
		int A_offset = Ty * TILE_SIZE
		int B_offset = Tx * TILE_SIZE
		for (int k = 0; k < TILE_SIZE; k++)
			s += As[A_offset + k] * Bs[B_offset + k];
		__syncthreads();
	}
	C[tidy * N + tidx] = s;
}

