#include "ew_op_gpu.h"
#include <stdio.h>
#include "CudaUtils.cuh"
#include <iostream>

// The kernel and the support functions taken form https://github.com/openai/blocksparse

template <typename V>
__global__ void __launch_bounds__(128) gemm_32x32x32_TN_vec4(float* U, const V* __restrict__ X, const V* __restrict__ E, uint C, uint K, uint N, uint C16, uint K16, uint inc_n, uint inc_c, uint inc_k)
{
    __shared__ float shrU[32*32*2 + 16*4];

    uint tid   = threadIdx.x;
    uint idx_C = blockIdx.y;
    uint idx_K = blockIdx.x;
    uint idx_N = blockIdx.z;

    uint tx = tid  & 7;
    uint ty = tid >> 3;
    uint  n = idx_N*32 + ty;

    // global offsets in vector units
    uint c = idx_C*8 + tx;
    uint k = idx_K*8 + tx;
    uint offsetC = n*C + c;
    uint offsetK = n*K + k;

    bool bc = c < C;
    bool bk = k < K;
    //bool bc = true;
    //bool bk = true;

    // shared offsets in bytes
    // When reading, each warp works on its own 8 rows.
    // These groups of 8 are added together at end.
    uint writeS = (ty*32 + tx*4) * 4;
    uint row8   = (tid & 96) * 32;
    uint readCs = row8 + (((tid & 16) >> 3) | (tid & 1)) * 16;
    uint readKs = row8 + ((tid >> 1) & 7) * 16;

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(writeS)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetC) : );
    asm("mov.b32 %0, %0;" : "+r"(offsetK) : );
    asm("mov.b32 %0, %0;" : "+r"(readCs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readKs)  : );

    // zero 32 accumulation registers
    float regU[8][4]; // [c][k]
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 4; k++)
            regU[c][k] = 0;

    // assume a minimum of one loop
    #pragma unroll 1
    do
    {
        V c00, c16;
        V k00, k16;
        ew_zero(c00); ew_zero(c16);
        ew_zero(k00); ew_zero(k16);
        const V* X00 = add_ptr_u(X, offsetC +   0);
        const V* X16 = add_ptr_u(X, offsetC + C16);
        const V* E00 = add_ptr_u(E, offsetK +   0);
        const V* E16 = add_ptr_u(E, offsetK + K16);
        if (bc)
        {
            c00 = __ldg(X00);
            c16 = __ldg(X16);
        }
        if (bk)
        {
            k00 = __ldg(E00);
            k16 = __ldg(E16);
        }
        offsetC += inc_c;
        offsetK += inc_k;
        n       += inc_n;

        __syncthreads();
        st_shared_v4(writeS + ( 0*32 + 0*16*32)*4, to_float(c00));
        st_shared_v4(writeS + ( 0*32 + 1*16*32)*4, to_float(c16));
        st_shared_v4(writeS + (32*32 + 0*16*32)*4, to_float(k00));
        st_shared_v4(writeS + (32*32 + 1*16*32)*4, to_float(k16));
        __syncthreads();

        float regC[8], regK[4];

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            // fetch outer product data
            ld_shared_v4(readCs + ( 0*32 + 32*j +  0)*4, &regC[0] );
            ld_shared_v4(readCs + ( 0*32 + 32*j + 16)*4, &regC[4] );
            ld_shared_v4(readKs + (32*32 + 32*j +  0)*4,  regK    );
            // compute outer product
            for (int c = 0; c < 8; c++)
                for (int k = 0; k < 4; k++)
                    regU[c][k] += regC[c] * regK[k];
        }
    } while (n < N);

    // conserve registers by forcing a reload of these
    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  ) :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_K) :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_C) :);

    // Arrange 4 tiles horizontally in the X direction: ((tid & 96) >> 2)
    // Add some spacing  to avoid write bank conflicts: (tidY << 2)
    int tidY = ((tid & 16) >> 3) | (tid & 1);
    int tidX = ((tid >> 1) & 7) + ((tid & 96) >> 2) + (tidY << 2);

    float4* storU4 = (float4*)&shrU[tidY*32*4*4 + tidX*4];

    __syncthreads();

    storU4[0*8*4] = *(float4*)regU[0];
    storU4[1*8*4] = *(float4*)regU[1];
    storU4[2*8*4] = *(float4*)regU[2];
    storU4[3*8*4] = *(float4*)regU[3];

    __syncthreads();

    // leaving vector math
    uint tid31 = tid & 31;
    uint tid32 = tid >> 5;
    C *= 4;
    K *= 4;

    float* readU = &shrU[tid32*32*4 + tid31];

    float u[4][4];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            u[j][i] = readU[j*32*4*4 + j*16 + i*32];

    // Tree reduce
    for (int k = 0; k < 4; k++)
        for (int j = 2; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                u[k][i] += u[k][i+j];

    k = idx_K*32 + tid31;
    c = idx_C*32 + tid32;
    bk = k < K;

    uint offsetU = c*K + k;
    atomicRed(add_ptr_u(U, offsetU +  0*K), u[0][0], 0, bk && c +  0 < C);
    atomicRed(add_ptr_u(U, offsetU +  4*K), u[1][0], 0, bk && c +  4 < C);
    atomicRed(add_ptr_u(U, offsetU +  8*K), u[2][0], 0, bk && c +  8 < C);
    atomicRed(add_ptr_u(U, offsetU + 12*K), u[3][0], 0, bk && c + 12 < C);
    //atomicRed(add_ptr_u(U, offsetU +  0*K), u[0][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU +  4*K), u[1][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU +  8*K), u[2][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU + 12*K), u[3][0], 0, true);

    __syncthreads();

    storU4[0*8*4] = *(float4*)regU[4];
    storU4[1*8*4] = *(float4*)regU[5];
    storU4[2*8*4] = *(float4*)regU[6];
    storU4[3*8*4] = *(float4*)regU[7];

    __syncthreads();

    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            u[j][i] = readU[j*32*4*4 + j*16 + i*32];

    // Tree reduce
    for (int k = 0; k < 4; k++)
        for (int j = 2; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                u[k][i] += u[k][i+j];

    atomicRed(add_ptr_u(U, offsetU + 16*K), u[0][0], 0, bk && c + 16 < C);
    atomicRed(add_ptr_u(U, offsetU + 20*K), u[1][0], 0, bk && c + 20 < C);
    atomicRed(add_ptr_u(U, offsetU + 24*K), u[2][0], 0, bk && c + 24 < C);
    atomicRed(add_ptr_u(U, offsetU + 28*K), u[3][0], 0, bk && c + 28 < C);
    //atomicRed(add_ptr_u(U, offsetU + 16*K), u[0][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU + 20*K), u[1][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU + 24*K), u[2][0], 0, true);
    //atomicRed(add_ptr_u(U, offsetU + 28*K), u[3][0], 0, true);
}

bool Gemm_TN(CUstream stream, uint SMs,
          float* u,
    const float* xf,
    const float* ef,
    uint C, uint K, uint N)
{
    //cuMemsetD32Async((CUdeviceptr)u, 0, C*K, stream);
    cudaMemset(u, 0, C*K*sizeof(float)); // TODO use cuMemsetD32Async instead of cudaMemset.

    const float4 *x = reinterpret_cast<const float4 *>(xf);
    const float4 *e = reinterpret_cast<const float4 *>(ef);

    uint gridK = CEIL_DIV(K, 32);
    uint gridC = CEIL_DIV(C, 32);
    uint gridN = CEIL_DIV(N, 32);
    C >>= 2;
    K >>= 2;

    // target mult of 6 blocks per SM
    uint smMult = 1, tiles = gridK*gridC;
         if (tiles == 1) smMult = 6;
    else if (tiles <= 4) smMult = 3;
    uint segments = SMs*smMult;
    if (segments > gridN)
        segments = gridN;
    uint seg_len = segments*32;

    dim3 grid(gridK, gridC, segments);
    gemm_32x32x32_TN_vec4<float4><<<grid,128,0,stream>>>(u, x, e, C, K, N, C*16, K*16, seg_len, seg_len*C, seg_len*K);
    return true; // TODO
}
