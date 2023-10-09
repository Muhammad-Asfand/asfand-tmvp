#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "SABLE_params.h"



#define WMMA_THREAD 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void convertnegacyclic (half *out, uint16_t *in);

__global__ void convertFp32ToU16modP (uint16_t *out, float *p1, float *p2, float *p3);

__global__ void convertU16ToFp16anticyclic(half *out, uint16_t *in);

__global__ void convertnegacyclictest(uint16_t *out, uint16_t *in);

__global__ void wmma_ker_padding2(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, float *c1, float *c2, float *c3);


__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in);

__global__ void submatrix(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, uint16_t *in1, uint16_t *in2);

__global__ void submatrix_cuda(uint16_t *a0, uint16_t *b0, uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *in1, uint16_t *in2);

__global__ void matvecp(uint16_t *A, uint16_t *x, uint16_t *p1, uint16_t *p2, uint16_t *p3);

__global__ void convertnegacyclictest2(uint16_t *out, uint16_t *in);

__global__ void matvecp_cuda(uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *a3, uint16_t *b3, uint16_t *p1, uint16_t *p2, uint16_t *p3);

__global__ void matvecout_cuda(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out);

__global__ void submatrix_m1(uint16_t *b1, uint16_t *b2, uint16_t *b3, uint16_t *in2);

__global__ void submatrix_m2(uint16_t *a1, uint16_t *a2, uint16_t *a3, uint16_t *in1);

__global__ void convertFp32ToU16modP_m (uint16_t *out, float *p1, float *p2, float *p3);

__global__ void matvecout_cudaq(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out);

__global__ void submatrix_m1_tensor(half *b1, half *b2, half *b3, uint16_t *in2);

__global__ void submatrix_m2_tensor(half *a1, half *a2, half *a3, uint16_t *in1);

__global__ void convertFp32ToU16modQ (uint16_t *out, float *in); 

__global__ void Mul_process(uint16_t *in);