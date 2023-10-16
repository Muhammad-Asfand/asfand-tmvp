
#include "../include/poly.cuh"
#include "../include/cuda_kernel.cuh"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <mma.h> 

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72
#define MODN(X) ((X) & (SABER_N-1))   
#define h1 (1 << (SABER_EQ - SABER_EP - 1))
#define h2 ((1 << (SABER_EP - 2)) - (1 << (SABER_EP - SABER_ET - 1)) + (1 << (SABER_EQ - SABER_EP - 1)))

#define MODP(X) ((X) & (SABER_P-1))
#define MODQ(X) ((X) & (SABER_Q-1))

using namespace nvcuda;

////////////////////////////////////////////////////////////////

__device__ int mod(int a, int b)
{

   int r = a & (b-1);
   return r < 0 ? r + b : r;
}

__global__ void packdp2av1(short2 *outa, uint16_t *a) 
{
    int i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    for(i=0; i<SABER_N/2; i++)
    {
      outa[bid * SABER_N/2 + i*SABER_N/2 + tid].x = a[bid * SABER_N + MODN(tid - i*2)]; 
      outa[bid * SABER_N/2 + i*SABER_N/2 + tid].y = a[bid * SABER_N + MODN(tid - i*2 - 1)];
    }    
}

__global__ void packdp2av2(short2 *outa1, short2 *outa2, uint16_t *a) 
{    
    int32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_L*SABER_N/2, bid2 = blockIdx.x*SABER_L*SABER_L*SABER_N;

   int i, j;
   
   for (i = 0; i < SABER_L; i++)
   {
      for (j = 0; j < SABER_L; j++)
      {     
         outa1[bid  + i*SABER_N*SABER_L/2 +j*SABER_N/2 + tid].x = a[bid2+ i*SABER_N*SABER_L+j*SABER_N + tid*2 + 1];  
         outa1[bid  + i*SABER_N*SABER_L/2 +j*SABER_N/2+ tid].y = a[bid2+ i*SABER_N*SABER_L+j*SABER_N + tid*2]; 
         outa2[bid  + i*SABER_N*SABER_L/2 +j*SABER_N/2+ tid].x = a[bid2+ i*SABER_N*SABER_L+j*SABER_N + (SABER_N - tid*2)%SABER_N];      
         outa2[bid  + i*SABER_N*SABER_L/2 +j*SABER_N/2+ tid].y = a[bid2+ i*SABER_N*SABER_L+j*SABER_N + (SABER_N - tid*2 - 1)%SABER_N ];           
      }
   }
}

__global__ void packdp2av2Inner(short2 *outa1, short2 *outa2, uint16_t *a) 
{    
    int32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_N/2, bid2 = blockIdx.x*SABER_L*SABER_N;

   int j;
   
     for (j = 0; j < SABER_L; j++)
      {     
         outa1[bid  + j*SABER_N/2 + tid].x = a[bid2+ j*SABER_N + tid*2 + 1];  
         outa1[bid  + j*SABER_N/2+ tid].y = a[bid2+ j*SABER_N + tid*2]; 
         outa2[bid  + j*SABER_N/2+ tid].x = a[bid2+ j*SABER_N + (SABER_N - tid*2)%SABER_N];      
         outa2[bid  + j*SABER_N/2+ tid].y = a[bid2+ j*SABER_N + (SABER_N - tid*2 - 1)%SABER_N ];
      }
}
__global__ void packdp2b(char4 *outb, uint16_t* b) 
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_N/2, bid2 = blockIdx.x*SABER_L*SABER_N;
   int j;
   
   for (j = 0; j < SABER_L; j++)
   { 
      outb[bid  +j*SABER_N/2 + tid].x = b[bid2+j*SABER_N + tid*2]; 
      outb[bid  +j*SABER_N/2 + tid].y = b[bid2+j*SABER_N + tid*2+1];    
   }  
}

// DoDP4Av4 with no C intrinsic, handle odd
__global__ void DoDP4Av7(uint16_t *out, short2 *a1, short2 *a2, char4* b) {
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_L*SABER_N/2, bid2 = blockIdx.x*SABER_L*SABER_N, bid3 = blockIdx.x*SABER_L*SABER_N/2;
    int i, j, k;
    int32_t sum1 = 0, sum2 = 0;
    ushort2 load_a;
    __shared__ ushort2 s_a1[SABER_N], s_a2[SABER_N];
    __shared__ char4 s_b[SABER_N];

   for (j = 0; j < SABER_L; j++)
   { 
       s_b[tid].x = b[bid3 +j*SABER_N/2 + tid].x;   
       s_b[tid].y = b[bid3 +j*SABER_N/2 + tid].y;
       // s_b[tid + SABER_N/2].x = b[bid3 +j*SABER_N/2 + tid + (SABER_N)/2].x;
       // s_b[tid + SABER_N/2].y = b[bid3 +j*SABER_N/2 + tid + (SABER_N)/2].y;      
      for (i = 0; i < SABER_L; i++)
      {
    
       s_a1[tid].x = a1[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid].x;
       s_a1[tid + (SABER_N)/2].x = a1[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid + (SABER_N)/2].x;
       s_a2[tid].x = a2[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid].x;
       s_a2[tid + (SABER_N)/2].x = a2[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid + (SABER_N)/2].x;

       s_a1[tid].y = a1[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid].y;
       s_a1[tid + (SABER_N)/2].y = a1[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid + (SABER_N)/2].y;
       s_a2[tid].y = a2[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid].y;
       s_a2[tid + (SABER_N)/2].y = a2[bid + i*SABER_N*SABER_L/2+j*SABER_N/2 + tid + (SABER_N)/2].y;
       __syncthreads();

       sum1 = 0;
       sum2 = 0;
       for(k=0; k<SABER_N/2; k++)
       { 
            // Processing even element
            load_a = s_a2[(tid*(SABER_N/2-1) + k)%(SABER_N/2)]; 
           if(k > tid)
           {               
               sum1 -= load_a.x*s_b[k].x + load_a.y * s_b[k].y;
               // if(tid==0) printf("-tid: %u %u += %u * %u + %u * %u\n", tid, sum1, load_a.x, s_b[k].x, load_a.y, s_b[k].y);     
           }
           else if (k==tid)
           {
               sum1 += load_a.x * s_b[k].x; 
               sum1 -= load_a.y * s_b[k].y; 
               // if(tid==0) printf("+*tid: %u %u += %u * %u - %u * %u\n", tid, sum1, load_a.x, s_b[k].x, load_a.y, s_b[k].y);    
           }
           else
           {
               sum1 += load_a.x*s_b[k].x + load_a.y * s_b[k].y; 
               // if(tid==0) printf("+tid: %u %u += %u * %u + %u * %u\n", tid, sum1, load_a.x, s_b[k].x, load_a.y, s_b[k].y);           
           }
           // Processing odd element
           load_a = s_a1[(tid + k*(SABER_N/2-1))%(SABER_N/2)];
           if(k<=tid)
           {               
               sum2 += load_a.x*s_b[k].x + load_a.y * s_b[k].y;  
               // if(tid==1) printf("+tid: %u %u += %u * %u + %u * %u\n", tid, sum2, load_a.x, s_b[k].x, load_a.y, s_b[k].y);  
           }
           else
           {
               sum2 -= load_a.x*s_b[k].x + load_a.y * s_b[k].y;  
               // if(tid==1) printf("-tid: %u %u += %u * %u + %u * %u\n", tid, sum2, load_a.x, s_b[k].x, load_a.y, s_b[k].y);  
           }
       }
       __syncthreads();
       // out[bid2 + i*SABER_N +tid*2] += MODQ(sum1);   
       // out[bid2 + i*SABER_N +tid*2+1] += MODQ(sum2);  
       out[bid2 + i*SABER_N +tid*2] += sum1;   
       out[bid2 + i*SABER_N +tid*2+1] += sum2;        
      }
   } 
}


__global__ void DoDP4Av7_inner(uint16_t *out, short2 *a1, short2 *a2, char4* b) {
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_N/2, bid2 = blockIdx.x*SABER_N, bid3 = blockIdx.x*SABER_L*SABER_N/2;
    int i, k;
    int32_t sum1 = 0, sum2 = 0;
    ushort2 load_a;
    __shared__ ushort2 s_a1[SABER_N], s_a2[SABER_N];
    __shared__ char4 s_b[SABER_N];

   for (i = 0; i < SABER_L; i++)
   {
      s_b[tid].x = b[bid3 +i*SABER_N/2 + tid].x;   
      s_b[tid].y = b[bid3 +i*SABER_N/2 + tid].y;
      s_a1[tid].x = a1[bid + i*SABER_N/2 + tid].x;
      s_a1[tid + SABER_N/2].x = a1[bid + i*SABER_N/2 + tid + (SABER_N)/2].x;
      s_a2[tid].x = a2[bid + i*SABER_N/2 + tid].x;
      s_a2[tid + SABER_N/2].x = a2[bid + i*SABER_N/2 + tid + (SABER_N)/2].x;

      s_a1[tid].y = a1[bid + i*SABER_N/2 + tid].y;
      s_a1[tid + SABER_N/2].y = a1[bid + i*SABER_N/2 + tid + SABER_N/2].y;
      s_a2[tid].y = a2[bid + i*SABER_N/2 + tid].y;
      s_a2[tid + SABER_N/2].y = a2[bid + i*SABER_N/2 + tid + SABER_N/2].y;
      __syncthreads();

      sum1 = 0;
      sum2 = 0;
      for(k=0; k<SABER_N/2; k++)
      { 
           if(k > tid)
           {
               load_a = s_a2[(tid*(SABER_N/2-1) + k)%(SABER_N/2)]; 
               sum1 -= load_a.x*s_b[k].x + load_a.y * s_b[k].y;    
           }
           else if (k==tid)
           {
               load_a = s_a2[(tid*(SABER_N/2-1) + k)%(SABER_N/2)];  
               sum1 += load_a.x * s_b[k].x; 
               sum1 -= load_a.y * s_b[k].y; 
           }
           else
           {
               load_a = s_a2[(tid*(SABER_N/2-1) + k)%(SABER_N/2)];  
               sum1 += load_a.x*s_b[k].x + load_a.y * s_b[k].y;    
           }
           
           if(k<=tid)
           {
               load_a = s_a1[(tid + k*(SABER_N/2-1))%(SABER_N/2)];
               sum2 += load_a.x*s_b[k].x + load_a.y * s_b[k].y;
           }
           else
           {
               load_a = s_a1[(tid + k*(SABER_N/2-1))%(SABER_N/2)];
               sum2 -= load_a.x*s_b[k].x + load_a.y * s_b[k].y; 
           }
       }
       __syncthreads();
       // out[bid2 + tid*2] += MODQ(sum1);   
       // out[bid2 + tid*2+1] += MODQ(sum2);  
       out[bid2 + tid*2] += sum1;   
       out[bid2 + tid*2+1] += sum2;        
      
   } 
}


__global__ void VecVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s)
{
   int16_t j, k, sum;
   uint32_t tid = threadIdx.x, bidx1 = blockIdx.x * SABER_N*SABER_L;
   uint32_t bidx2 = blockIdx.x * SABER_N;
   __shared__ int16_t s0[SABER_L*SABER_N], a[SABER_N];
   for (j = 0; j < SABER_L; j++)
   {
      a[tid] = g_a[bidx1  + j*SABER_N + tid];
      s0[tid] = g_s[bidx1 + j*SABER_N + tid];   
      __syncthreads();
      sum = 0;// use register to accumulate
      for(k=0; k<tid+1; k++)
         sum += s0[tid-k] * a[k];  
      for(k=1; k<SABER_N-tid; k++)
         sum -= s0[tid+k] * a[(SABER_N)-k];   
       __syncthreads();
      // r[bidx2 + tid] +=MODQ(sum) ;       
       r[bidx2 + tid] +=sum ;     
   }
  
 }
__global__ void MatVecMul_gpu_shared2(uint16_t *r, uint16_t *g_a, uint16_t *g_s)
{
   int16_t k, sum;
   uint32_t tid = threadIdx.x, bidx1 = blockIdx.x * SABER_N*SABER_L*SABER_L;
   uint32_t bidx2 = blockIdx.x * SABER_N*SABER_L;
   __shared__ int16_t s0[SABER_N], a[SABER_N];
   
   a[tid] = g_a[bidx1 + tid];
   s0[tid] = g_s[bidx2 + tid];   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
    __syncthreads();
   // r[bidx2 + tid] +=MODQ(sum) ;       
    r[bidx2 + tid] +=sum ;       
 }

// Only for Saber, not for FireSaber or LightSaber
__global__ void MatVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s)
{
   uint16_t k, sum;
   uint32_t tid = threadIdx.x, bidx1 = blockIdx.x * SABER_N*SABER_L*SABER_L;
   uint32_t bidx2 = blockIdx.x * SABER_N*SABER_L;
   __shared__ uint16_t s0[SABER_N], s1[SABER_N], s2[SABER_N], a[SABER_N];

   // i=0, j=0
   a[tid] = g_a[bidx1 + tid];
   s0[tid] = g_s[bidx2 + tid];   
   __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + tid] +=sum ;    
   __syncthreads();

    // i=0, j=1
   a[tid] = g_a[bidx1 + SABER_N + tid];
   s1[tid] = g_s[bidx2 + SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s1[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s1[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + tid] +=sum ;
   __syncthreads();

    // i=0, j=2
   a[tid] = g_a[bidx1 + 2*SABER_N + tid];
   s2[tid] = g_s[bidx2 + 2*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s2[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s2[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + tid] +=sum ;
   __syncthreads();

    // i=1, j=0
   a[tid] = g_a[bidx1 + SABER_L*SABER_N + tid];   
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=1, j=1
   a[tid] = g_a[bidx1 + SABER_L*SABER_N + SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s1[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s1[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=1, j=2
   a[tid] = g_a[bidx1 + SABER_L*SABER_N + 2*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s2[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s2[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=0
   a[tid] = g_a[bidx1 + 2*SABER_L*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=1
   a[tid] = g_a[bidx1 + 2*SABER_L*SABER_N + SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s1[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s1[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=2
   a[tid] = g_a[bidx1 + 2*SABER_L*SABER_N + 2*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s2[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s2[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
}

/////////////////////////////////////////////////////////////////////////////////////


__global__ void post_process(uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x * SABER_N*SABER_L; 
   int i;
   for (i = 0; i < SABER_L; i++)
   {
      in[bid + i*SABER_N + tid] = (in[bid +i*SABER_N + tid] + h1) >> (SABER_EQ - SABER_EP);
   }
}

__global__ void post_process2(uint16_t *out, uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N; 
   out[bid + tid] = (out[bid + tid] - (in[bid + tid] << (SABER_EP - 1)) + h1) >> (SABER_EP - SABER_ET);
}


__global__ void post_process3(uint16_t *out, uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N; 

   out[bid + tid] = (out[bid + tid] + h2 - (in[bid + tid] << (SABER_EP - SABER_ET))) >> (SABER_EP - 1);
}

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64 - offset)))

__device__ static uint64_t load64(const unsigned char *x)
{
  unsigned long long r = 0, i;

  for (i = 0; i < 8; ++i)
  {
    r |= (unsigned long long)x[i] << 8 * i;
  }
  return r;
}

__device__ static void store64(uint8_t *x, uint64_t u)
{
  unsigned int i;

  for (i = 0; i < 8; ++i)
  {
    x[i] = u;
    u >>= 8;
  }
}

__constant__ const uint64_t KeccakF_RoundConstants[NROUNDS] =
    {
        (uint64_t)0x0000000000000001ULL,
        (uint64_t)0x0000000000008082ULL,
        (uint64_t)0x800000000000808aULL,
        (uint64_t)0x8000000080008000ULL,
        (uint64_t)0x000000000000808bULL,
        (uint64_t)0x0000000080000001ULL,
        (uint64_t)0x8000000080008081ULL,
        (uint64_t)0x8000000000008009ULL,
        (uint64_t)0x000000000000008aULL,
        (uint64_t)0x0000000000000088ULL,
        (uint64_t)0x0000000080008009ULL,
        (uint64_t)0x000000008000000aULL,
        (uint64_t)0x000000008000808bULL,
        (uint64_t)0x800000000000008bULL,
        (uint64_t)0x8000000000008089ULL,
        (uint64_t)0x8000000000008003ULL,
        (uint64_t)0x8000000000008002ULL,
        (uint64_t)0x8000000000000080ULL,
        (uint64_t)0x000000000000800aULL,
        (uint64_t)0x800000008000000aULL,
        (uint64_t)0x8000000080008081ULL,
        (uint64_t)0x8000000000008080ULL,
        (uint64_t)0x0000000080000001ULL,
        (uint64_t)0x8000000080008008ULL};

__device__ static void KeccakF1600_StatePermute(uint64_t *state)
{
  int round;

  uint64_t Aba, Abe, Abi, Abo, Abu;
  uint64_t Aga, Age, Agi, Ago, Agu;
  uint64_t Aka, Ake, Aki, Ako, Aku;
  uint64_t Ama, Ame, Ami, Amo, Amu;
  uint64_t Asa, Ase, Asi, Aso, Asu;
  uint64_t BCa, BCe, BCi, BCo, BCu;
  uint64_t Da, De, Di, Do, Du;
  uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
  uint64_t Ega, Ege, Egi, Ego, Egu;
  uint64_t Eka, Eke, Eki, Eko, Eku;
  uint64_t Ema, Eme, Emi, Emo, Emu;
  uint64_t Esa, Ese, Esi, Eso, Esu;

  //copyFromState(A, state)
  Aba = state[0];
  Abe = state[1];
  Abi = state[2];
  Abo = state[3];
  Abu = state[4];
  Aga = state[5];
  Age = state[6];
  Agi = state[7];
  Ago = state[8];
  Agu = state[9];
  Aka = state[10];
  Ake = state[11];
  Aki = state[12];
  Ako = state[13];
  Aku = state[14];
  Ama = state[15];
  Ame = state[16];
  Ami = state[17];
  Amo = state[18];
  Amu = state[19];
  Asa = state[20];
  Ase = state[21];
  Asi = state[22];
  Aso = state[23];
  Asu = state[24];

  for (round = 0; round < NROUNDS; round += 2)
  {
    //    prepareTheta
    BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    //thetaRhoPiChiIotaPrepareTheta(round  , A, E)
    Da = BCu ^ ROL(BCe, 1);
    De = BCa ^ ROL(BCi, 1);
    Di = BCe ^ ROL(BCo, 1);
    Do = BCi ^ ROL(BCu, 1);
    Du = BCo ^ ROL(BCa, 1);

    Aba ^= Da;
    BCa = Aba;
    Age ^= De;
    BCe = ROL(Age, 44);
    Aki ^= Di;
    BCi = ROL(Aki, 43);
    Amo ^= Do;
    BCo = ROL(Amo, 21);
    Asu ^= Du;
    BCu = ROL(Asu, 14);
    Eba = BCa ^ ((~BCe) & BCi);
    Eba ^= (uint64_t)KeccakF_RoundConstants[round];
    Ebe = BCe ^ ((~BCi) & BCo);
    Ebi = BCi ^ ((~BCo) & BCu);
    Ebo = BCo ^ ((~BCu) & BCa);
    Ebu = BCu ^ ((~BCa) & BCe);

    Abo ^= Do;
    BCa = ROL(Abo, 28);
    Agu ^= Du;
    BCe = ROL(Agu, 20);
    Aka ^= Da;
    BCi = ROL(Aka, 3);
    Ame ^= De;
    BCo = ROL(Ame, 45);
    Asi ^= Di;
    BCu = ROL(Asi, 61);
    Ega = BCa ^ ((~BCe) & BCi);
    Ege = BCe ^ ((~BCi) & BCo);
    Egi = BCi ^ ((~BCo) & BCu);
    Ego = BCo ^ ((~BCu) & BCa);
    Egu = BCu ^ ((~BCa) & BCe);

    Abe ^= De;
    BCa = ROL(Abe, 1);
    Agi ^= Di;
    BCe = ROL(Agi, 6);
    Ako ^= Do;
    BCi = ROL(Ako, 25);
    Amu ^= Du;
    BCo = ROL(Amu, 8);
    Asa ^= Da;
    BCu = ROL(Asa, 18);
    Eka = BCa ^ ((~BCe) & BCi);
    Eke = BCe ^ ((~BCi) & BCo);
    Eki = BCi ^ ((~BCo) & BCu);
    Eko = BCo ^ ((~BCu) & BCa);
    Eku = BCu ^ ((~BCa) & BCe);

    Abu ^= Du;
    BCa = ROL(Abu, 27);
    Aga ^= Da;
    BCe = ROL(Aga, 36);
    Ake ^= De;
    BCi = ROL(Ake, 10);
    Ami ^= Di;
    BCo = ROL(Ami, 15);
    Aso ^= Do;
    BCu = ROL(Aso, 56);
    Ema = BCa ^ ((~BCe) & BCi);
    Eme = BCe ^ ((~BCi) & BCo);
    Emi = BCi ^ ((~BCo) & BCu);
    Emo = BCo ^ ((~BCu) & BCa);
    Emu = BCu ^ ((~BCa) & BCe);

    Abi ^= Di;
    BCa = ROL(Abi, 62);
    Ago ^= Do;
    BCe = ROL(Ago, 55);
    Aku ^= Du;
    BCi = ROL(Aku, 39);
    Ama ^= Da;
    BCo = ROL(Ama, 41);
    Ase ^= De;
    BCu = ROL(Ase, 2);
    Esa = BCa ^ ((~BCe) & BCi);
    Ese = BCe ^ ((~BCi) & BCo);
    Esi = BCi ^ ((~BCo) & BCu);
    Eso = BCo ^ ((~BCu) & BCa);
    Esu = BCu ^ ((~BCa) & BCe);

    //    prepareTheta
    BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
    Da = BCu ^ ROL(BCe, 1);
    De = BCa ^ ROL(BCi, 1);
    Di = BCe ^ ROL(BCo, 1);
    Do = BCi ^ ROL(BCu, 1);
    Du = BCo ^ ROL(BCa, 1);

    Eba ^= Da;
    BCa = Eba;
    Ege ^= De;
    BCe = ROL(Ege, 44);
    Eki ^= Di;
    BCi = ROL(Eki, 43);
    Emo ^= Do;
    BCo = ROL(Emo, 21);
    Esu ^= Du;
    BCu = ROL(Esu, 14);
    Aba = BCa ^ ((~BCe) & BCi);
    Aba ^= (uint64_t)KeccakF_RoundConstants[round + 1];
    Abe = BCe ^ ((~BCi) & BCo);
    Abi = BCi ^ ((~BCo) & BCu);
    Abo = BCo ^ ((~BCu) & BCa);
    Abu = BCu ^ ((~BCa) & BCe);

    Ebo ^= Do;
    BCa = ROL(Ebo, 28);
    Egu ^= Du;
    BCe = ROL(Egu, 20);
    Eka ^= Da;
    BCi = ROL(Eka, 3);
    Eme ^= De;
    BCo = ROL(Eme, 45);
    Esi ^= Di;
    BCu = ROL(Esi, 61);
    Aga = BCa ^ ((~BCe) & BCi);
    Age = BCe ^ ((~BCi) & BCo);
    Agi = BCi ^ ((~BCo) & BCu);
    Ago = BCo ^ ((~BCu) & BCa);
    Agu = BCu ^ ((~BCa) & BCe);

    Ebe ^= De;
    BCa = ROL(Ebe, 1);
    Egi ^= Di;
    BCe = ROL(Egi, 6);
    Eko ^= Do;
    BCi = ROL(Eko, 25);
    Emu ^= Du;
    BCo = ROL(Emu, 8);
    Esa ^= Da;
    BCu = ROL(Esa, 18);
    Aka = BCa ^ ((~BCe) & BCi);
    Ake = BCe ^ ((~BCi) & BCo);
    Aki = BCi ^ ((~BCo) & BCu);
    Ako = BCo ^ ((~BCu) & BCa);
    Aku = BCu ^ ((~BCa) & BCe);

    Ebu ^= Du;
    BCa = ROL(Ebu, 27);
    Ega ^= Da;
    BCe = ROL(Ega, 36);
    Eke ^= De;
    BCi = ROL(Eke, 10);
    Emi ^= Di;
    BCo = ROL(Emi, 15);
    Eso ^= Do;
    BCu = ROL(Eso, 56);
    Ama = BCa ^ ((~BCe) & BCi);
    Ame = BCe ^ ((~BCi) & BCo);
    Ami = BCi ^ ((~BCo) & BCu);
    Amo = BCo ^ ((~BCu) & BCa);
    Amu = BCu ^ ((~BCa) & BCe);

    Ebi ^= Di;
    BCa = ROL(Ebi, 62);
    Ego ^= Do;
    BCe = ROL(Ego, 55);
    Eku ^= Du;
    BCi = ROL(Eku, 39);
    Ema ^= Da;
    BCo = ROL(Ema, 41);
    Ese ^= De;
    BCu = ROL(Ese, 2);
    Asa = BCa ^ ((~BCe) & BCi);
    Ase = BCe ^ ((~BCi) & BCo);
    Asi = BCi ^ ((~BCo) & BCu);
    Aso = BCo ^ ((~BCu) & BCa);
    Asu = BCu ^ ((~BCa) & BCe);
  }

  //copyToState(state, A)
  state[0] = Aba;
  state[1] = Abe;
  state[2] = Abi;
  state[3] = Abo;
  state[4] = Abu;
  state[5] = Aga;
  state[6] = Age;
  state[7] = Agi;
  state[8] = Ago;
  state[9] = Agu;
  state[10] = Aka;
  state[11] = Ake;
  state[12] = Aki;
  state[13] = Ako;
  state[14] = Aku;
  state[15] = Ama;
  state[16] = Ame;
  state[17] = Ami;
  state[18] = Amo;
  state[19] = Amu;
  state[20] = Asa;
  state[21] = Ase;
  state[22] = Asi;
  state[23] = Aso;
  state[24] = Asu;

#undef round
}

#include <string.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__device__ static void keccak_absorb(uint64_t *s,
                          unsigned int r,
                          const unsigned char *m, unsigned long long int mlen,
                          unsigned char p)
{
  unsigned long long i;
  unsigned char t[200];

  while (mlen >= r)
  {
    for (i = 0; i < r / 8; ++i)
      s[i] ^= load64(m + 8 * i);

    KeccakF1600_StatePermute(s);
    mlen -= r;
    m += r;
  }

  for (i = 0; i < r; ++i)
    t[i] = 0;
  for (i = 0; i < mlen; ++i)
    t[i] = m[i];
  t[i] = p;
  t[r - 1] |= 128;
  for (i = 0; i < r / 8; ++i)
    s[i] ^= load64(t + 8 * i);
}

__device__ static void keccak_squeezeblocks(unsigned char *h, unsigned long long int nblocks,
                                 uint64_t *s,
                                 unsigned int r)
{
  unsigned int i;
  while (nblocks > 0)
  {
    KeccakF1600_StatePermute(s);
    for (i = 0; i < (r >> 3); i++)
    {
      store64(h + 8 * i, s[i]);
    }
    h += r;
    nblocks--;
  }
}
__device__ void shake128(unsigned char *output, unsigned long long outlen,
              const unsigned char *input, unsigned long long inlen)
{
  uint64_t s[25];
  unsigned char t[SHAKE128_RATE];
  unsigned long long nblocks = outlen / SHAKE128_RATE;
  size_t i;

  for (i = 0; i < 25; ++i)
    s[i] = 0;

  /* Absorb input */
  keccak_absorb(s, SHAKE128_RATE, input, inlen, 0x1F);

  /* Squeeze output */
  keccak_squeezeblocks(output, nblocks, s, SHAKE128_RATE);

  output += nblocks * SHAKE128_RATE;
  outlen -= nblocks * SHAKE128_RATE;

  if (outlen)
  {
    keccak_squeezeblocks(t, 1, s, SHAKE128_RATE);
    for (i = 0; i < outlen; i++)
      output[i] = t[i];
  }
}

__device__  void BS2POLq(const uint8_t bytes[SABER_POLYBYTES], uint16_t *data)
{
   size_t j, offset_byte, offset_data;
   for (j = 0; j < SABER_N / 8; j++)
   {
      offset_byte = 13 * j;
      offset_data = 8 * j;
      data[offset_data + 0] = (bytes[offset_byte + 0] & (0xff)) | ((bytes[offset_byte + 1] & 0x1f) << 8);
      data[offset_data + 1] = (bytes[offset_byte + 1] >> 5 & (0x07)) | ((bytes[offset_byte + 2] & 0xff) << 3) | ((bytes[offset_byte + 3] & 0x03) << 11);
      data[offset_data + 2] = (bytes[offset_byte + 3] >> 2 & (0x3f)) | ((bytes[offset_byte + 4] & 0x7f) << 6);
      data[offset_data + 3] = (bytes[offset_byte + 4] >> 7 & (0x01)) | ((bytes[offset_byte + 5] & 0xff) << 1) | ((bytes[offset_byte + 6] & 0x0f) << 9);
      data[offset_data + 4] = (bytes[offset_byte + 6] >> 4 & (0x0f)) | ((bytes[offset_byte + 7] & 0xff) << 4) | ((bytes[offset_byte + 8] & 0x01) << 12);
      data[offset_data + 5] = (bytes[offset_byte + 8] >> 1 & (0x7f)) | ((bytes[offset_byte + 9] & 0x3f) << 7);
      data[offset_data + 6] = (bytes[offset_byte + 9] >> 6 & (0x03)) | ((bytes[offset_byte + 10] & 0xff) << 2) | ((bytes[offset_byte + 11] & 0x07) << 10);
      data[offset_data + 7] = (bytes[offset_byte + 11] >> 3 & (0x1f)) | ((bytes[offset_byte + 12] & 0xff) << 5);
   }
}


__device__ void BS2POLVECq(const uint8_t bytes[SABER_POLYVECBYTES], uint16_t *data)
{
   size_t i;
   for (i = 0; i < SABER_L; i++)
   {
      BS2POLq(bytes + i * SABER_POLYBYTES, data + i*SABER_N);
   }
}

__device__ uint64_t load_littleendian(const uint8_t *x, int bytes)
{
  int i;
  uint64_t r = x[0];
  for (i = 1; i < bytes; i++)
    r |= (uint64_t)x[i] << (8 * i);
  return r;
}

__device__ void cbd_gpu(uint16_t s[SABER_N], const uint8_t buf[SABER_POLYCOINBYTES])
{
#if SABER_MU == 6
  uint32_t t, d, a[4], b[4];
  int i, j;

  for (i = 0; i < SABER_N / 4; i++)
  {
    t = load_littleendian(buf + 3 * i, 3);
    d = 0;
    for (j = 0; j < 3; j++)
      d += (t >> j) & 0x249249;

    a[0] = d & 0x7;
    b[0] = (d >> 3) & 0x7;
    a[1] = (d >> 6) & 0x7;
    b[1] = (d >> 9) & 0x7;
    a[2] = (d >> 12) & 0x7;
    b[2] = (d >> 15) & 0x7;
    a[3] = (d >> 18) & 0x7;
    b[3] = (d >> 21);

    s[4 * i + 0] = (uint16_t)(a[0] - b[0]);
    s[4 * i + 1] = (uint16_t)(a[1] - b[1]);
    s[4 * i + 2] = (uint16_t)(a[2] - b[2]);
    s[4 * i + 3] = (uint16_t)(a[3] - b[3]);
  }
#elif SABER_MU == 8 
  uint32_t t, d, a[4], b[4];
  int i, j;

  for (i = 0; i < SABER_N / 4; i++)
  {
    t = load_littleendian(buf + 4 * i, 4);
    d = 0;
    for (j = 0; j < 4; j++)
      d += (t >> j) & 0x11111111;

    a[0] = d & 0xf;
    b[0] = (d >> 4) & 0xf;
    a[1] = (d >> 8) & 0xf;
    b[1] = (d >> 12) & 0xf;
    a[2] = (d >> 16) & 0xf;
    b[2] = (d >> 20) & 0xf;
    a[3] = (d >> 24) & 0xf;
    b[3] = (d >> 28);

    s[4 * i + 0] = (uint16_t)(a[0] - b[0]);
    s[4 * i + 1] = (uint16_t)(a[1] - b[1]);
    s[4 * i + 2] = (uint16_t)(a[2] - b[2]);
    s[4 * i + 3] = (uint16_t)(a[3] - b[3]);
  }
#elif SABER_MU == 10
  uint64_t t, d, a[4], b[4];
  int i, j;

  for (i = 0; i < SABER_N / 4; i++)
  {
    t = load_littleendian(buf + 5 * i, 5);
    d = 0;
    for (j = 0; j < 5; j++)
      d += (t >> j) & 0x0842108421UL;

    a[0] = d & 0x1f;
    b[0] = (d >> 5) & 0x1f;
    a[1] = (d >> 10) & 0x1f;
    b[1] = (d >> 15) & 0x1f;
    a[2] = (d >> 20) & 0x1f;
    b[2] = (d >> 25) & 0x1f;
    a[3] = (d >> 30) & 0x1f;
    b[3] = (d >> 35);

    s[4 * i + 0] = (uint16_t)(a[0] - b[0]);
    s[4 * i + 1] = (uint16_t)(a[1] - b[1]);
    s[4 * i + 2] = (uint16_t)(a[2] - b[2]);
    s[4 * i + 3] = (uint16_t)(a[3] - b[3]);
  }
#else
#error "Unsupported SABER parameter."
#endif
}

__global__ void shake128_gpu(uint8_t *out, const uint8_t *in, size_t inlen, uint32_t outlen, uint32_t out_stride) 
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint8_t p = 0x1F;   // For absorb
    uint32_t r = 168;   // For shake128
    int const s = threadIdx.x % 5;
    __shared__ uint64_t A[25];
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];
    __shared__ uint64_t d_data[25];
    uint32_t i, count=0;
    __shared__ uint8_t t[200];
    uint32_t nblocks = outlen / SHAKE128_RATE;
    outlen -= nblocks * SHAKE128_RATE;  // Remain one block?

    // Initialize arrays to zeros
    for (i = 0; i < 8; ++i) {
        if (tid < 25) 
        {
            t[i*25 + tid] = 0;
        }
    }
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; d_data[tid] = 0; 
    }

    // Absorb phase
    while (inlen >= r) 
    {         
        if(tid<17) d_data[tid] ^= load64(in + bid*SABER_INDCPA_PUBLICKEYBYTES + 8 * tid + count*r); //136 / 8
        if (tid < 25) {
              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
        }        
        inlen -= r;
        count++;
    }

    if(tid==0) 
    {
        t[inlen] = p;
        t[r - 1] |= 128;
    }
    __syncthreads();
    // printf("%u %x\n", tid, d_data[tid]);
    uint32_t repeat = (inlen/blockDim.x)+1;
    if(repeat==0) repeat = 1;
    for (i = 0; i < repeat; i++)
    {
        if(tid < inlen) t[i*blockDim.x + tid] = in[i*blockDim.x + tid+ count*r];
        inlen-=blockDim.x;
        __syncthreads();
    }   
    
    if(tid < 21) d_data[tid] ^= load64(t + 8*tid);    
    // if(bid==0) if(tid < 25) printf("%u %lx %u\n", tid, d_data[tid], count);
    // if(tid < 25) printf("%u %d %u\n", tid, in[tid], inlen);
   // if(threadIdx.x==0) 
   // {
   //    printf("\n seed\n"); for (int i = 0; i < SABER_SEEDBYTES; i++) printf("%u ", in[i]);  
   // }

    // Squeeze phase
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; 
    }    
    count=0;
    while (nblocks > 0) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
            store64(out + bid*out_stride+ count*r + 8*tid, d_data[ tid]);       
        }                
        count++;
        nblocks--;
    }

    if (outlen) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];         
        }             
        
        if(tid<outlen/8) store64(out + bid*out_stride+ count*r + 8*tid, d_data[tid]);
    }
}
// void GenMatrix(uint16_t A[SABER_L][SABER_L][SABER_N], const uint8_t seed[SABER_SEEDBYTES])
__global__ void GenMatrix_gpu(uint16_t *A, uint8_t *seed)
{
   uint8_t buf[SABER_L * SABER_POLYVECBYTES];
   int i;
   uint32_t tid = threadIdx.x;
   // printf("GenMatrix: %u\n", SABER_L * SABER_POLYVECBYTES);
   shake128(buf, sizeof(buf), seed + tid*SABER_SEEDBYTES, SABER_SEEDBYTES);

   for (i = 0; i < SABER_L; i++)
   {
      BS2POLVECq(buf + i * SABER_POLYVECBYTES, A + tid*SABER_L*SABER_L*SABER_N + i*SABER_L*SABER_N);
   }
}

__global__ void GenMatrix_gpu2(uint16_t *A, uint8_t *seed)
{
   uint8_t buf[SABER_L * SABER_POLYVECBYTES];
   int i;
   uint32_t tid = threadIdx.x;
   // printf("GenMatrix: %u\n", SABER_L * SABER_POLYVECBYTES);
   shake128(buf, sizeof(buf), seed + tid*SABER_INDCPA_PUBLICKEYBYTES, SABER_SEEDBYTES);

   for (i = 0; i < SABER_L; i++)
   {
      BS2POLVECq(buf + i * SABER_POLYVECBYTES, A + tid*SABER_L*SABER_L*SABER_N + i*SABER_L*SABER_N);
   }
}

// void GenSecret_gpu(uint16_t s[SABER_L][SABER_N], const uint8_t seed[SABER_NOISE_SEEDBYTES])
__global__ void GenSecret_gpu(uint16_t *s, uint8_t *seed)
{
   uint8_t buf[SABER_L * SABER_POLYCOINBYTES] = {0};
   size_t i;
   uint32_t tid = threadIdx.x;
   shake128(buf, sizeof(buf), seed + tid*64, SABER_NOISE_SEEDBYTES);

   for (i = 0; i < SABER_L; i++)
   {
      cbd_gpu(s + tid*SABER_N*SABER_L + i*SABER_N, buf + i * SABER_POLYCOINBYTES);
   }
}

// void sha3_256(unsigned char *output, const unsigned char *input, unsigned long long inlen)
__global__ void sha3_256_gpu(uint8_t *output, uint8_t *input, unsigned long long inlen, uint32_t in_stride, uint32_t out_stride)
{
  uint64_t s[25];
  unsigned char t[SHA3_256_RATE];
  size_t i;
   uint32_t tid = threadIdx.x;

  for (i = 0; i < 25; ++i)
    s[i] = 0;

  /* Absorb input */
  keccak_absorb(s, SHA3_256_RATE, input + tid*in_stride, inlen, 0x06);

  /* Squeeze output */
  keccak_squeezeblocks(t, 1, s, SHA3_256_RATE);

  for (i = 0; i < 32; i++)
    output[i + tid*out_stride] = t[i];
}

// void sha3_512(unsigned char *output, const unsigned char *input, unsigned long long inlen)
__global__ void sha3_512_gpu(uint8_t *output, uint8_t *input, unsigned long long inlen)
{
  uint64_t s[25];
  unsigned char t[SHA3_512_RATE];
  size_t i;
   uint32_t tid = threadIdx.x;
  for (i = 0; i < 25; ++i)
    s[i] = 0;

  /* Absorb input */
  keccak_absorb(s, SHA3_512_RATE, input + tid*64, inlen, 0x06);

  /* Squeeze output */
  keccak_squeezeblocks(t, 1, s, SHA3_512_RATE);

  for (i = 0; i < 64; i++)
    output[i + tid*64] = t[i];
}

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

/////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////

__global__ void convertFp32ToU16modP (uint16_t *out, float *p1, float *p2, float *p3) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P

   uint32_t tidx = threadIdx.x;
   uint32_t bidx = blockIdx.x*SABER_N;
   uint32_t bidx2 = blockIdx.x*SABER_N/2;
    
   out[bidx + tidx] +=  (int32_t) (p1[tidx] + p2[tidx]);
   out[bidx + (SABER_N/2) + tidx] += (int32_t) (p1[tidx] - p3[tidx]); 

}

//////////////////////////////////////////////////////////////

__global__ void convertU16ToFp16Negate(half *out, uint16_t *in) {
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   uint16_t temp = in[bidx*SABER_N*SABER_L + tidx];
   if(temp == 2047) out[bidx * (SABER_N) + tidx] = -1;
   else out[bidx * (SABER_N) + tidx] = temp;    
  
}

////////////////////////////////////////////////////////////////

__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   int32_t idx = tidx-bidx;

   if(idx<0)
      out[bidx + tidx*SABER_N] = in[mod(idx, SABER_N)] * (-1);

   else
      out[bidx + tidx*SABER_N] = in[mod(idx, SABER_N)];
}

/////////////////////////////////////////////////////////////////

__global__ void wmma_ker_padding(half *a, half *b, float *c) 
{
    // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> c_frag;

   // Each warp compute 16 elements along index i
   uint32_t warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   uint32_t ldA_offset, ldB_offset, row_idx, col_idx, st_offset;
   
   row_idx = warpID%((SABER_N)/WMMA_M)*WMMA_M;
   col_idx = warpID/((SABER_N)/WMMA_M)*WMMA_M;   
    
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   for (int i = 0; i < (SABER_N)/WMMA_M; i ++)    
    {
      ldA_offset = row_idx*(SABER_N) + i*WMMA_M;
      ldB_offset = col_idx*(SABER_N) + i*WMMA_M;
    
      wmma::load_matrix_sync(a_frag, a + ldA_offset , SABER_N);    
      wmma::load_matrix_sync(b_frag, b + ldB_offset , SABER_N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    st_offset = row_idx + col_idx * SABER_N;
    
    wmma::store_matrix_sync(c + st_offset, c_frag, SABER_N, wmma::mem_col_major);  
}

__global__ void convertFp32ToU16modQ (uint16_t *out, float *in) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P
   int tidx = threadIdx.x;
   int bidx = blockIdx.x* SABER_N; 

   out[bidx*SABER_L +  tidx] += MODQ(((int32_t) in[tidx]));
 
}   

//////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////

__global__ void matvecout_cuda(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out)
{
   int tidx = threadIdx.x;
   int bidx2 = blockIdx.x*SABER_N;
   

   out[bidx2 + tidx] +=  MODP(p1[bidx2 + tidx*2] + p1[bidx2 + tidx*2 + 1] + p2[bidx2 + tidx*2] + p2[bidx2 + tidx*2 + 1]);
   out[bidx2 + (SABER_N/2) + tidx] += MODP(p1[bidx2 + tidx*2] + p1[bidx2 + tidx*2 + 1] - p3[bidx2 + tidx*2] - p3[bidx2 + tidx*2 + 1]);
 
}

///////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////

__global__ void submatrix_m2(uint16_t *a1, uint16_t *a2, uint16_t *a3, uint16_t *in1)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N;
   

   a1[bidx + tidx] = in1[bidx2 + tidx];
   
   a2[bidx + tidx] = in1[bidx2 + (SABER_N/2) + tidx] - in1[bidx2 + tidx];
   
   a3[bidx + tidx] = in1[bidx2 + tidx] - in1[bidx2 + tidx + (SABER_N/2*SABER_N)];

}

__global__ void submatrix_m1(uint16_t *b1, uint16_t *b2, uint16_t *b3, uint16_t *in2)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N*SABER_L;
   //uint16_t temp1 = 0, temp2 = 0, temp3 = 0;

   b1[bidx + tidx] = in2[bidx2 + tidx] + in2[bidx2 + SABER_N/2 + tidx];
   
   b2[bidx + tidx] = in2[bidx2 + tidx + (SABER_N/2)];
   
   b3[bidx + tidx] = in2[bidx2 + tidx];

}

__global__ void matvecout_cudaq(uint16_t *p1, uint16_t *p2, uint16_t *p3, uint16_t *out)
{
   int tidx = threadIdx.x;
   int bidx2 = blockIdx.x*SABER_N*SABER_L;
   int bidx = blockIdx.x*SABER_N;

   out[bidx2 + tidx] +=  MODQ(p1[bidx + tidx*2] + p1[bidx + tidx*2 + 1] + p2[bidx + tidx*2] + p2[bidx + tidx*2 + 1]);
   out[bidx2 + (SABER_N/2) + tidx] += MODQ(p1[bidx + tidx*2] + p1[bidx + tidx*2 + 1] - p3[bidx + tidx*2] - p3[bidx + tidx*2 + 1]);

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

__global__ void submatrix_m2_tensor(half *a1, half *a2, half *a3, uint16_t *in1)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x*SABER_N/2;
   int bidx2 = blockIdx.x*SABER_N*SABER_L;



   a1[bidx + tidx] = in1[bidx2 + tidx] + in1[bidx2 + SABER_N/2 + tidx];
   
   a2[bidx + tidx] = in1[tidx + (SABER_N/2)];
   
   a3[bidx + tidx] = in1[tidx];

}

__global__ void convertFp32ToU16modP_m (uint16_t *out, float *p1, float *p2, float *p3) 
{   
   // Move the output (32-bit) of Tensor-Core to (16-bit) GPU with mod-P

   int32_t tidx = threadIdx.x;
   int32_t bidx = blockIdx.x*SABER_N*SABER_L;
   //int32_t bidx2 = blockIdx.x*SABER_N/2;


   out[bidx + tidx] +=  MODQ((int32_t) (p1[tidx] + p2[tidx]));
   out[bidx + (SABER_N/2) + tidx] += MODQ((int32_t) (p1[tidx] - p3[tidx]));

}

__global__ void Mul_process(uint16_t *in)
{

   int tidx = threadIdx.x;
   int bidx = blockIdx.x;
   uint16_t temp1 = 0;

   temp1 = in[bidx*SABER_N*SABER_L + tidx];
   if(temp1 == 8191) in[bidx*SABER_N*SABER_L + tidx] = -1;
   else in[bidx*SABER_N*SABER_L + tidx] = MODP(temp1);
}