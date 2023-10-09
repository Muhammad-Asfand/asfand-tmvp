#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mma.h>

#include "../include/poly.cuh"
#include "../include/SABLE_params.h"

#define MODP(X) ((X) & (SABER_P-1))
#define MODQ(X) ((X) & (SABER_Q-1))

using namespace nvcuda;

////////////////////////////////////////////////////////////////

__device__ int mod(int a, int b)
{

   int r = a & (b-1);
   return r < 0 ? r + b : r;
}

__device__ int mod1(int a, int b)
{

   int r = a % b;
   return r < 0 ? r + b : r;
}

////////////////////////////////////////////////////////////////

__global__ void convertnegacyclic(half *out, uint16_t *in)
{
   // To move pack the Polynomial (a) in nega-clockwise direction
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;
   //uint16_t temp[SABER_N*SABER_N];

   int32_t idx = 2*tidx-bidx;

   if(idx<0)
      out[bidx + tidx*SABER_N] = MODP(in[mod(idx, SABER_N)] * (-1));

   else
      out[bidx + tidx*SABER_N] = in[mod(idx, SABER_N)];
/*
   for (int i = 0; i < SABER_N; i++)
   {
      out[tidx + SABER_N*i] = temp[2*SABER_N*i + tidx];
   }
*/
}
 

 //////////////////////////////////////////////////////////////////

__global__ void convertU16ToFp16anticyclic(half *out, uint16_t *in) { 
   // To move pack the Polynomial (b) in nega-anti-clockwise direction

   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   //out[bidx*(4) + tidx ] = in[mod1(tidx + bidx, 4)]; 

   int32_t idx = tidx+bidx;

   if(idx>SABER_N-1)
      out[bidx*(SABER_N) + tidx] = in[mod1(idx, SABER_N)] * (-1);

   else
      out[bidx*(SABER_N) + tidx] = in[mod1(idx, SABER_N)];
}
 

///////////////////////////////////////////////////////////////////

__global__ void convertFp32ToU16modP (uint16_t *out, float *p1, float *p2, float *p3) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P

   uint32_t tidx = threadIdx.x;
   uint32_t bidx = blockIdx.x*SABER_N;
   //uint32_t bidx2 = blockIdx.x*SABER_N/2;
    
   out[bidx + tidx] +=  (int32_t) (p1[tidx] + p2[tidx]);
   out[bidx + (SABER_N/2) + tidx] += (int32_t) (p1[tidx] - p3[tidx]); 

}

__global__ void convertFp32ToU16modP_m (uint16_t *out, float *p1, float *p2, float *p3) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P

   int32_t tidx = threadIdx.x;
   int32_t bidx = blockIdx.x*SABER_N*SABER_K;
   //int32_t bidx2 = blockIdx.x*SABER_N/2;


   out[bidx + tidx] +=  MODP((int32_t) (p1[tidx] + p2[tidx]));
   out[bidx + (SABER_N/2) + tidx] += MODP((int32_t) (p1[tidx] - p3[tidx]));

}

/////////////////////////////////////////////////////////////////

__global__ void convertnegacyclictest(uint16_t *out, uint16_t *in)
{
   // To move pack the Polynomial (a) in nega-clockwise direction
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   int idx = tidx-bidx;

   if(idx<0)
      out[bidx + tidx*256] = MODP(in[mod(idx, 256)] * (-1));

   else
      out[bidx + tidx*256] = in[mod(idx, 256)];
}
///////////////////////////////////////////////////////////////////////////
__global__ void convertnegacyclictest2(uint16_t *out, uint16_t *in)
{
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   int32_t idx = tidx-bidx;

   if(idx<0)
      out[bidx + tidx*SABER_N] = MODQ(in[mod(idx, SABER_N)] * (-1));

   else
      out[bidx + tidx*SABER_N] = in[mod(idx, SABER_N)];
}

//////////////////////////////////////////////////////////////

__global__ void submatrix(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, uint16_t *in1, uint16_t *in2)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N;
   

   a1[bidx + tidx] = in1[bidx2 + tidx];
   b1[bidx + tidx] = in2[tidx] + in2[SABER_N/2 + tidx];


   a2[bidx + tidx] = in1[bidx2 + (SABER_N/2) + tidx] - in1[bidx2 + tidx];
   b2[bidx + tidx] = in2[tidx + (SABER_N/2)];



   a3[bidx + tidx] = in1[bidx2 + tidx] - in1[bidx2 + tidx + (SABER_N/2*SABER_N)];
   b3[bidx + tidx] = in2[tidx];

}

__global__ void submatrix_m1(uint16_t *b1, uint16_t *b2, uint16_t *b3, uint16_t *in2)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N*SABER_K;
   //uint16_t temp1 = 0, temp2 = 0, temp3 = 0;

   b1[bidx + tidx] = in2[bidx2 + tidx] + in2[bidx2 + SABER_N/2 + tidx];
   
   b2[bidx + tidx] = in2[bidx2 + tidx + (SABER_N/2)];
   
   b3[bidx + tidx] = in2[bidx2 + tidx];

}

__global__ void submatrix_m2(uint16_t *a1, uint16_t *a2, uint16_t *a3, uint16_t *in1)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N;
   

   a1[bidx + tidx] = in1[bidx2 + tidx];
   
   a2[bidx + tidx] = in1[bidx2 + (SABER_N/2) + tidx] - in1[bidx2 + tidx];
   
   a3[bidx + tidx] = in1[bidx2 + tidx] - in1[bidx2 + tidx + (SABER_N/2*SABER_N)];

}

__global__ void submatrix_m1_tensor(half *b1, half *b2, half *b3, uint16_t *in2)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N;
   

   b1[bidx + tidx] = in2[bidx2 + tidx];
   
   b2[bidx + tidx] = in2[bidx2 + (SABER_N/2) + tidx] - in2[bidx2 + tidx];
   
   b3[bidx + tidx] = in2[bidx2 + tidx] - in2[bidx2 + tidx + (SABER_N/2*SABER_N)];

}

////////////////////////////////////////////////////////////////////////////

__global__ void Mul_process(uint16_t *in)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x;
   uint16_t temp1 = 0;

   temp1 = in[bidx*SABER_N*SABER_K + tidx];
   if(temp1 == 2047) in[bidx*SABER_N*SABER_K + tidx] = -1;
   else in[bidx*SABER_N*SABER_K + tidx] = MODP(temp1);
}

//////////////////////////////////////////////////////////////////////////////////////

__global__ void submatrix_m2_tensor(half *a1, half *a2, half *a3, uint16_t *in1)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N*SABER_K;



   a1[bidx + tidx] = in1[bidx2 + tidx] + in1[bidx2 + SABER_N/2 + tidx];
   
   a2[bidx + tidx] = in1[tidx + (SABER_N/2)];
   
   a3[bidx + tidx] = in1[tidx];

}


///////////////////////////////////////////////////////////////

__global__ void submatrix_cuda(uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *a3, uint16_t *b3, uint16_t *in1, uint16_t *in2)
{

   int32_t tidx = threadIdx.x;
   int32_t bidx = blockIdx.x*SABER_N/2;
   int32_t bidx2 = blockIdx.x*SABER_N;
   

   a1[bidx + tidx] = in1[bidx2 + tidx];
   b1[bidx + tidx] = in2[tidx] + in2[SABER_N/2 + tidx];

   a2[bidx + tidx] = in1[bidx2 + (SABER_N/2) + tidx] - in1[bidx2 + tidx];
   b2[bidx + tidx] = in2[tidx + (SABER_N/2)];

   a3[bidx + tidx] = in1[bidx2 + tidx] - in1[bidx2 + tidx + (SABER_N/2*SABER_N)];
   b3[bidx + tidx] = in2[tidx];
}



__global__ void matvecp(uint16_t *A, uint16_t *x, uint16_t *p1, uint16_t *p2, uint16_t *p3)
{
   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;

   uint16_t sum1 = 0, sum2 = 0, sum3 = 0;
  
   //p1[tidx] = 0;
   //p2[tidx] = 0;
   //p3[tidx] = 0;
   
   __shared__ uint16_t a[SABER_N/2 * SABER_N/2], b[SABER_N/2];
   
   //p1//////////////////
   b[tidx] = x[tidx] + x[SABER_N/2 + tidx];
   for (int i = 0; i < (SABER_N/2); i++)
   {
     a[tidx + i*(SABER_N/2)] = A[tidx + i*(SABER_N/2)];
   }

   __syncthreads();
        
   for (int j = 0; j < (SABER_N/2); j++) {
      sum1 += a[tidx * (SABER_N/2) + j] * b[j];
   }
   p1[bidx + tidx] = sum1;
   __syncthreads();
   //p2///////////////////

   for (int i = 0; i < (SABER_N/2); i++)
   {
      a[tidx + i*(SABER_N/2)] = A[(SABER_N/2*SABER_N/2) + tidx + i*(SABER_N/2)] - A[tidx + i*(SABER_N/2)];
   }
   b[tidx] = x[tidx + (SABER_N/2)];

   __syncthreads();

   for (int j = 0; j < (SABER_N/2); j++) {
      sum2 += a[tidx * (SABER_N/2) + j] * b[j];
   }
   p2[bidx + tidx] = sum2;
   
   __syncthreads();
   //p3///////////////////

   for (int i = 0; i < (SABER_N/2); i++)
   {
      a[tidx + i*(SABER_N/2)] = A[tidx + i*(SABER_N/2)] - A[(SABER_N*SABER_N/2) + tidx + i*(SABER_N/2)];
   }
   b[tidx] = x[tidx];

   __syncthreads();

   for (int j = 0; j < (SABER_N/2); j++) {
      sum3 += a[tidx * (SABER_N/2) + j] * b[j];
   }
   p3[bidx + tidx] = sum3;


}

__global__ void matvecp_cuda(uint16_t *a1, uint16_t *b1, uint16_t *a2, uint16_t *b2, uint16_t *a3, uint16_t *b3, uint16_t *p1, uint16_t *p2, uint16_t *p3)
{

  
   int32_t tidx = threadIdx.x;
   int32_t bidx = blockIdx.x;
   int i,j;

   uint16_t sum1 = 0, sum2 = 0, sum3 = 0;

   __shared__ uint16_t a[SABER_N/2 * SABER_N/2], b[SABER_N];
   

   for (i = 0; i < (SABER_N/4); i++)
   {
      a[tidx + i*(SABER_N)] = a1[tidx + i*(SABER_N)];
   }
   b[tidx] = b1[tidx];


   __syncthreads();

   for (j = 0; j < (SABER_N/4); j++)
   {
      sum1 += a[tidx*SABER_N/4 + j] * b[(tidx % 2) * SABER_N/4 + j];
   }
   p1[bidx*SABER_N + tidx] = sum1;

   __syncthreads();

   // p2///////////////////

   for (i = 0; i < (SABER_N/4); i++)
   {
     a[tidx + i*(SABER_N)] = a2[tidx + i*(SABER_N)];
   }
   b[tidx] = b2[tidx];

   __syncthreads();

   for (j = 0; j < (SABER_N/4); j++)
   {
      sum2 += a[tidx*SABER_N/4 + j] * b[(tidx % 2) * SABER_N/4 + j];
   }
   p2[bidx*SABER_N + tidx] = sum2;

   __syncthreads();

   // p3///////////////////

   for (i = 0; i < (SABER_N/4); i++)
   {
      a[tidx + i*(SABER_N)] = a3[tidx + i*(SABER_N)];
   }
   b[tidx] = b3[tidx];

   __syncthreads();

   for (j = 0; j < (SABER_N/4); j++)
   {
      sum3 += a[tidx*SABER_N/4 + j] * b[(tidx % 2) * SABER_N/4 + j];
   }
   p3[bidx*SABER_N + tidx] = sum3;

}

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void matvecout_cuda(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out)
{
   int tidx = threadIdx.x;
   int bidx2 = blockIdx.x*SABER_N;
   //int idx = SABER_N/2 - tidx - 1;
   
   //out[bidx2 + tidx] +=  MODP(p1[tidx*8] + p1[tidx*8 + 1] + p1[tidx*8 + 2] + p1[tidx*8 + 3] + p1[tidx*8 + 4] + p1[tidx*8 + 5] + p1[tidx*8 + 6] + p1[tidx*8 + 7] + p2[tidx*8] + p2[tidx*8 + 1] + p2[tidx*8 + 2] + p2[tidx*8 + 3] + p2[tidx*8 + 4] + p2[tidx*8 + 5] + p2[tidx*8 + 6] + p2[tidx*8 + 7]);

   //out[bidx2 + (SABER_N/2) + tidx] += MODP(p1[tidx*8] + p1[tidx*8 + 1] + p1[tidx*8 + 2] + p1[tidx*8 + 3] + p1[tidx*8 + 4] + p1[tidx*8 + 5] + p1[tidx*8 + 6] + p1[tidx*8 + 7] - p3[tidx*8] - p3[tidx*8 + 1] - p3[tidx*8 + 2] - p3[tidx*8 + 3] - p3[tidx*8 + 4] - p3[tidx*8 + 5] - p3[tidx*8 + 6] - p3[tidx*8 + 7]); 


   out[bidx2 + tidx] +=  MODP(p1[bidx2 + tidx*2] + p1[bidx2 + tidx*2 + 1] + p2[bidx2 + tidx*2] + p2[bidx2 + tidx*2 + 1]);
   out[bidx2 + (SABER_N/2) + tidx] += MODP(p1[bidx2 + tidx*2] + p1[bidx2 + tidx*2 + 1] - p3[bidx2 + tidx*2] - p3[bidx2 + tidx*2 + 1]);
 
}

__global__ void matvecout_cudaq(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out)
{
   int tidx = threadIdx.x;
   int bidx2 = blockIdx.x*SABER_N*SABER_K;
   int bidx = blockIdx.x*SABER_N;

   out[bidx2 + tidx] +=  MODQ(p1[bidx + tidx*2] + p1[bidx + tidx*2 + 1] + p2[bidx + tidx*2] + p2[bidx + tidx*2 + 1]);
   out[bidx2 + (SABER_N/2) + tidx] += MODQ(p1[bidx + tidx*2] + p1[bidx + tidx*2 + 1] - p3[bidx + tidx*2] - p3[bidx + tidx*2 + 1]);
}

////////////////////////////////////////////////////////

__global__ void wmma_ker_padding2(half *a1, half *b1, half *a2, half *b2, half *a3, half *b3, float *c1, float *c2, float *c3) 
{
    // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, half, wmma::row_major> a1_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, half, wmma::col_major> b1_frag;
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, half, wmma::row_major> a2_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, half, wmma::col_major> b2_frag;
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, half, wmma::row_major> a3_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, half, wmma::col_major> b3_frag;

   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> x_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> y_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> z_frag;

   // Each warp compute 16 elements along index i
   uint32_t warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   uint32_t ldA_offset, ldB_offset, row_idx, col_idx, st_offset;
   
   row_idx = warpID%((SABER_N/2)/WMMA_M)*WMMA_M;
   col_idx = warpID/((SABER_N/2)/WMMA_M)*WMMA_M;   
    
   // Initialize the output to zero
   wmma::fill_fragment(x_frag, 0.0f);
   wmma::fill_fragment(y_frag, 0.0f);
   wmma::fill_fragment(z_frag, 0.0f);

   for (int i = 0; i < (SABER_N/2)/WMMA_M; i++)    
   {
      ldA_offset = row_idx*(SABER_N/2) + i*WMMA_M;
      ldB_offset = col_idx*(SABER_N/2) + i*WMMA_M;
    
      wmma::load_matrix_sync(a1_frag, a1 + ldA_offset , SABER_N/2);    
      wmma::load_matrix_sync(b1_frag, b1 + ldB_offset , SABER_N/2);
      wmma::mma_sync(x_frag, a1_frag, b1_frag, x_frag);
   }

   st_offset = row_idx + col_idx * SABER_N/2;
   wmma::store_matrix_sync(c1 + st_offset, x_frag, SABER_N/2, wmma::mem_col_major);

   
   for (int i = 0; i < (SABER_N/2)/WMMA_M; i ++)    
   {
      ldA_offset = row_idx*(SABER_N/2) + i*WMMA_M;
      ldB_offset = col_idx*(SABER_N/2) + i*WMMA_M;
    
      wmma::load_matrix_sync(a2_frag, a2 + ldA_offset , SABER_N/2);    
      wmma::load_matrix_sync(b2_frag, b2 + ldB_offset , SABER_N/2);
      wmma::mma_sync(y_frag, a2_frag, b2_frag, y_frag);
   }

   st_offset = row_idx + col_idx * SABER_N/2;
   wmma::store_matrix_sync(c2 + st_offset, y_frag, SABER_N/2, wmma::mem_col_major);


   for (int i = 0; i < (SABER_N/2)/WMMA_M; i ++)    
   {
      ldA_offset = row_idx*(SABER_N/2) + i*WMMA_M;
      ldB_offset = col_idx*(SABER_N/2) + i*WMMA_M;
    
      wmma::load_matrix_sync(a3_frag, a3 + ldA_offset , SABER_N/2);    
      wmma::load_matrix_sync(b3_frag, b3 + ldB_offset , SABER_N/2);
      wmma::mma_sync(z_frag, a3_frag, b3_frag, z_frag);
   }

   st_offset = row_idx + col_idx * SABER_N/2;
   wmma::store_matrix_sync(c3 + st_offset, z_frag, SABER_N/2, wmma::mem_col_major);

}   

//////////////////////////////////////////////////////////////////////////////

__global__ void convertFp32ToU16modQ (uint16_t *out, float *in) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P
   int tidx = threadIdx.x;
   int bidx = blockIdx.x* SABER_N;
   int32_t temp; 

   temp = (int32_t) in[bidx + tidx];
   out[bidx*SABER_K +  tidx] += MODQ(temp);
   //out[bidx*SABER_K +  tidx] += MODQ(((int32_t) in[bidx + tidx]));
 
}   

///////////////////////////////////////////////////////////////////

__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   int32_t idx = tidx-bidx;

   if(idx<0)
      out[bidx + tidx*SABER_N] = MODQ(in[mod1(idx, SABER_N)] * (-1));

   else
      out[bidx + tidx*SABER_N] = in[mod1(idx, SABER_N)];
}