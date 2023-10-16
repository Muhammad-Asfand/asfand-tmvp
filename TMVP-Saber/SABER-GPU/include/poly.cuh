#include "../include/params.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define WMMA_THREAD 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define MODQ(X) ((X) & (SABER_Q-1))

__global__ void MatVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s);
__global__ void MatVecMul_gpu_shared2(uint16_t *r, uint16_t *g_a, uint16_t *g_s);
__global__ void convertu16tos16(int16_t *b, uint16_t *a);
__global__ void convertu16tos16negate(int16_t *b, uint16_t *a);
__global__ void MatVecMul_gpu_dp2a(uint16_t *r, uint16_t *g_a, uint16_t *g_s);
__global__ void packdp2av2(short2 *outa1, short2 *outa2, uint16_t *a) ;
__global__ void packdp2b(char4 *outb, uint16_t* b) ;
__global__ void DoDP4Av7(uint16_t *out, short2 *a1, short2 *a2, char4* b) ;
__global__ void DoDP4Av7_inner(uint16_t *out, short2 *a1, short2 *a2, char4* b);
__global__ void packdp2av2Inner(short2 *outa1, short2 *outa2, uint16_t *a) ;
__global__ void post_process(uint16_t *in);
__global__ void post_process2(uint16_t *out, uint16_t *in);
__global__ void VecVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s);
__global__ void post_process2(uint16_t *out, uint16_t *in);
__global__ void DoDP4Av7_inner2(uint16_t *out, short2 *a1, short2 *a2, char4* b) ;
__global__ void packdp2av2Inner2(short2 *outa1, short2 *outa2, uint16_t *a) ;
__global__ void packdp2b2(char4 *outb, uint16_t* b)  ;
__global__ void post_process3(uint16_t *out, uint16_t *in);
__global__ void GenMatrix_gpu(uint16_t *A, uint8_t *seed);
__global__ void GenMatrix_gpu2(uint16_t *A, uint8_t *seed);
__global__ void GenSecret_gpu(uint16_t *s, uint8_t *seed);
__global__ void sha3_256_gpu(uint8_t *output, uint8_t *input, unsigned long long inlen, uint32_t in_stride, uint32_t out_stride);
__global__ void sha3_512_gpu(uint8_t *output, uint8_t *input, unsigned long long inlen);


#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) /* works on the GPU also for 
b = 64 or c = 64 */
#define NROUNDS 24
#define ROL(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

__constant__ uint64_t rc[5][NROUNDS] = {
    { 0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL },
    { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL } };

/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
Only the first entry of each pair is a rho-offset. The second part is
used in the R64 macros. */
__constant__ int ro[25][2] = {
    /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
    /*x=0*/{ 0,64 }, /*x=1*/{ 44,20 }, /*x=2*/{ 43,21 }, /*x=3*/{ 21,43 }, /*x=4*/{ 14,50 },
    /*x=1*/{ 1,63 }, /*x=2*/{ 6,58 }, /*x=3*/{ 25,39 }, /*x=4*/{ 8,56 }, /*x=0*/{ 18,46 },
    /*x=2*/{ 62, 2 }, /*x=3*/{ 55, 9 }, /*x=4*/{ 39,25 }, /*x=0*/{ 41,23 }, /*x=1*/{ 2,62 },
    /*x=3*/{ 28,36 }, /*x=4*/{ 20,44 }, /*x=0*/{ 3,61 }, /*x=1*/{ 45,19 }, /*x=2*/{ 61, 3 },
    /*x=4*/{ 27,37 }, /*x=0*/{ 36,28 }, /*x=1*/{ 10,54 }, /*x=2*/{ 15,49 }, /*x=3*/{ 56, 8 } };

__constant__ int a[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23 };

__constant__ int b[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3 };

__constant__ int c[25][3] = {
    { 0, 1, 2 },{ 1, 2, 3 },{ 2, 3, 4 },{ 3, 4, 0 },{ 4, 0, 1 },
    { 5, 6, 7 },{ 6, 7, 8 },{ 7, 8, 9 },{ 8, 9, 5 },{ 9, 5, 6 },
    { 10,11,12 },{ 11,12,13 },{ 12,13,14 },{ 13,14,10 },{ 14,10,11 },
    { 15,16,17 },{ 16,17,18 },{ 17,18,19 },{ 18,19,15 },{ 19,15,16 },
    { 20,21,22 },{ 21,22,23 },{ 22,23,24 },{ 23,24,20 },{ 24,20,21 } };

__constant__ int d[25] = {
    0,  1,  2,  3,  4,
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    5,  6,  7,  8,  9,
    15, 16, 17, 18, 19 };


// __constant__ uint32_t a[25];
// __constant__ uint32_t b[25];
// __constant__ uint32_t c[25][3];
// __constant__ uint32_t d[25];
// __constant__ uint32_t ro[25][2];
// __constant__ uint64_t rc[5][NROUNDS];

__global__ void shake128_gpu(uint8_t *out, const uint8_t *in, size_t inlen, uint32_t outlen, uint32_t out_stride) ;

__global__ void convertnegacyclictest(uint16_t *out, uint16_t *in);

__global__ void submatrix(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, uint16_t *in1, uint16_t *in2);

__global__ void wmma_ker_padding2(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, float *c1, float *c2, float *c3);

__global__ void convertFp32ToU16modP (uint16_t *out, float *p1, float *p2, float *p3);

__global__ void convertU16ToFp16Negate(half *out, uint16_t *in);

__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in);

__global__ void wmma_ker_padding(half *a, half *b, float *c);

__global__ void convertFp32ToU16modQ (uint16_t *out, float *in);

__global__ void submatrix_cuda(uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *a3, uint16_t *b3, uint16_t *in1, uint16_t *in2);

__global__ void matvecp_cuda(uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *a3, uint16_t *b3, uint16_t *p1, uint16_t *p2, uint16_t *p3);

__global__ void matvecout_cuda(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out);

__global__ void convertnegacyclictest2(uint16_t *out, uint16_t *in);

__global__ void submatrix_m2(uint16_t *a1, uint16_t *a2, uint16_t *a3, uint16_t *in1);

__global__ void submatrix_m1(uint16_t *b1, uint16_t *b2, uint16_t *b3, uint16_t *in2);

__global__ void matvecout_cudaq(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out);

__global__ void submatrix_m1_tensor(half *b1, half *b2, half *b3, uint16_t *in2);

__global__ void submatrix_m2_tensor(half *a1, half *a2, half *a3, uint16_t *in1);

__global__ void convertFp32ToU16modP_m (uint16_t *out, float *p1, float *p2, float *p3);

__global__ void Mul_process(uint16_t *in);