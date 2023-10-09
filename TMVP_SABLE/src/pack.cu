#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/pack.cuh"
#include "../include/SABLE_params.h"

#define SCHB_N 16

#define N_RES (SABER_N << 1)
#define N_SB (SABER_N >> 2)
#define N_SB_RES (2*N_SB-1)

#define h1 (1 << (SABER_EQ - SABER_EP - 1))
#define h2 ((1 << (SABER_EP - 2)) - (1 << (SABER_EP - SABER_ET - 1)) + (1 << (SABER_EQ - SABER_EP - 1)))

#define MODP(X) ((X) & (SABER_P-1))

//////////////////////////////////////////////////////////////

__device__ static void BS2POLq_gpu2(uint8_t *bytes, uint16_t *data)
{
    size_t offset_byte, offset_data;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_K*SABER_POLYVECBYTES, bid2 =blockIdx.x*SABER_N*SABER_K*SABER_K ;
    offset_byte = bid + 11 * tid;
    offset_data = bid2 + 8 * tid;

        data[offset_data + 0]= bytes[ offset_byte + 0 ] |  ((bytes[ offset_byte + 1 ] & 0x07)<<8);
        data[offset_data + 1]= ( (bytes[ offset_byte + 1 ]>>3) & (0x1f)) |  ((bytes[ offset_byte + 2 ] & 0x3f)<<5);
        data[offset_data + 2]= ( (bytes[ offset_byte + 2 ]>>6) & (0x03)) |  ((bytes[ offset_byte + 3 ] & 0xff)<<2) |  ((bytes[ offset_byte + 4 ] & 0x01)<<10);
        data[offset_data + 3]= ( (bytes[ offset_byte + 4 ]>>1) & (0x7f)) |  ((bytes[ offset_byte + 5 ] & 0x0f)<<7);
        data[offset_data + 4]= ( (bytes[ offset_byte + 5 ]>>4) & (0x0f)) |  ((bytes[ offset_byte + 6 ] & 0x7f)<<4);
        data[offset_data + 5]= ( (bytes[ offset_byte + 6 ]>>7) & (0x01)) |  ((bytes[ offset_byte + 7 ] & 0xff)<<1) |  ((bytes[ offset_byte + 8 ] & 0x03)<<9);
        data[offset_data + 6]= ( (bytes[ offset_byte + 8 ]>>2) & (0x3f)) |  ((bytes[ offset_byte + 9 ] & 0x1f)<<6);
        data[offset_data + 7]= ( (bytes[ offset_byte + 9 ]>>5) & (0x07)) |  ((bytes[ offset_byte + 10 ] & 0xff)<<3); 
}

////////////////////////////////////////////////////////////// 
    
__global__ void BS2POLVECq_gpu2(uint8_t *bytes, uint16_t *data)
{
    size_t i, j;

    for (i = 0; i < SABER_K; i++)
    {
        for (j = 0; j < SABER_K; j++)
            BS2POLq_gpu2(bytes + (i*SABER_K + j) * SABER_EQ*SABER_N/8, data + i*SABER_K*SABER_N + j*SABER_N);
    }
}

///////////////////////////////////////////////////////////////

__global__ void MatVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s)
{
   uint16_t k, sum;
   uint32_t tid = threadIdx.x, bidx1 = blockIdx.x * SABER_N*SABER_K*SABER_K;
   uint32_t bidx2 = blockIdx.x * SABER_N*SABER_K;
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
   a[tid] = g_a[bidx1 + SABER_K*SABER_N + tid];   
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=1, j=1
   a[tid] = g_a[bidx1 + SABER_K*SABER_N + SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s1[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s1[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=1, j=2
   a[tid] = g_a[bidx1 + SABER_K*SABER_N + 2*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s2[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s2[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=0
   a[tid] = g_a[bidx1 + 2*SABER_K*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s0[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s0[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=1
   a[tid] = g_a[bidx1 + 2*SABER_K*SABER_N + SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s1[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s1[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
   __syncthreads();

    // i=2, j=2
   a[tid] = g_a[bidx1 + 2*SABER_K*SABER_N + 2*SABER_N + tid];
    __syncthreads();   
   sum = 0;// use register to accumulate
   for(k=0; k<tid+1; k++)
      sum += s2[tid-k] * a[k];  
   for(k=1; k<SABER_N-tid; k++)
      sum -= s2[tid+k] * a[(SABER_N)-k];   
   r[bidx2 + 2*SABER_N + tid] +=sum ;
   __syncthreads();
}

///////////////////////////////////////////////////////////////

__global__ void post_process(uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x * SABER_N*SABER_K; 
   uint16_t mod_q=SABER_Q-1;
   int i;

   for (i = 0; i < SABER_K; i++)
   {
      in[bid + i*SABER_N + tid] = (in[bid +i*SABER_N + tid] + h1) & mod_q;
      in[bid + i*SABER_N + tid] = (in[bid +i*SABER_N + tid] >> (SABER_EQ - SABER_EP));
   }
}

///////////////////////////////////////////////////////////////

__device__ void POLp2BS_gpu(uint8_t *bytes, uint16_t *data)
{
    size_t offset_byte, offset_data;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x*SABER_N*SABER_K;

    
    offset_byte = 9 * tid + bid;
    offset_data = 8 * tid + bid2;

    bytes[offset_byte + 0] = ( data[ offset_data + 0 ] & (0xff));
    bytes[offset_byte + 1] = ( (data[ offset_data + 0 ] >>8) & 0x01 ) | ((data[ offset_data + 1 ] & 0x7f) << 1);
    bytes[offset_byte + 2] = ( (data[ offset_data + 1 ] >>7) & 0x03 ) | ((data[ offset_data + 2 ] & 0x3f) << 2);
    bytes[offset_byte + 3] = ( (data[ offset_data + 2 ] >>6) & 0x07 ) | ((data[ offset_data + 3 ] & 0x1f) << 3);
    bytes[offset_byte + 4] = ( (data[ offset_data + 3 ] >>5) & 0x0f ) | ((data[ offset_data + 4 ] & 0x0f) << 4);
    bytes[offset_byte + 5] = ( (data[ offset_data + 4 ] >>4) & 0x1f ) | ((data[ offset_data + 5 ] & 0x07) << 5);
    bytes[offset_byte + 6] = ( (data[ offset_data + 5 ] >>3) & 0x3f ) | ((data[ offset_data + 6 ] & 0x03) << 6);
    bytes[offset_byte + 7] = ( (data[ offset_data + 6 ] >>2) & 0x7f ) | ((data[ offset_data + 7 ] & 0x01) << 7);
    bytes[offset_byte + 8] = ( (data[ offset_data + 7 ] >>1) & 0xff );

}

///////////////////////////////////////////////////////////////

__global__ void POLVECp2BS_gpu(uint8_t *bytes, uint16_t *data)
{
    size_t i;

    for (i = 0; i < SABER_K; i++)
    {
        POLp2BS_gpu(bytes + i * (9 * SABER_N / 8), data + i*SABER_N);
    }
}

///////////////////////////////////////////////////////////////

__device__ void BS2POLp_gpu(const uint8_t bytes[SABER_POLYVECCOMPRESSEDBYTES], uint16_t data[SABER_N], uint32_t stride)
{
    size_t offset_byte, offset_data;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*stride, bid2 = blockIdx.x*SABER_N*SABER_K;

    offset_byte = 9 * tid + bid;
    offset_data = 8 * tid + bid2;

    data[offset_data + 0]= (bytes[ offset_byte + 0 ] & (0xff)) |  ((bytes[ offset_byte + 1 ] & 0x01)<<8);
    data[offset_data + 1]= ( (bytes[ offset_byte + 1 ]>>1) & (0x7f)) |  ((bytes[ offset_byte + 2 ] & 0x03)<<7);  
    data[offset_data + 2]= ( (bytes[ offset_byte + 2 ]>>2) & (0x3f)) |  ((bytes[ offset_byte + 3 ] & 0x07)<<6);
    data[offset_data + 3]= ( (bytes[ offset_byte + 3 ]>>3) & (0x1f)) |  ((bytes[ offset_byte + 4 ] & 0x0f)<<5);
    data[offset_data + 4]= ( (bytes[ offset_byte + 4 ]>>4) & (0x0f)) |  ((bytes[ offset_byte + 5 ] & 0x1f)<<4);
    data[offset_data + 5]= ( (bytes[ offset_byte + 5 ]>>5) & (0x07)) |  ((bytes[ offset_byte + 6 ] & 0x3f)<<3);
    data[offset_data + 6]= ( (bytes[ offset_byte + 6 ]>>6) & (0x03)) |  ((bytes[ offset_byte + 7 ] & 0x7f)<<2);
    data[offset_data + 7]= ( (bytes[ offset_byte + 7 ]>>7) & (0x01)) |  ((bytes[ offset_byte + 8 ] & 0xff)<<1);
}

///////////////////////////////////////////////////////////////

__global__ void BS2POLVECp_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride)
{
    size_t i;
    for (i = 0; i < SABER_K; i++)
    {
        BS2POLp_gpu(bytes + i * (9 * SABER_N / 8), data + i*SABER_N, stride);
    }
}

///////////////////////////////////////////////////////////////

__global__ void post_process2(uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x * SABER_N*SABER_K; 
   //uint16_t mod_p=SABER_P-1;
   int i;

   for (i = 0; i < SABER_K; i++)
   {
      in[bid + i*SABER_N + tid] = MODP(in[bid +i*SABER_N + tid]);
    }
}

///////////////////////////////////////////////////////////////

__global__ void VecVecMul_Inner_gpu(uint16_t *r, uint16_t *g_a, uint16_t *g_s)
{
   int16_t j, k, sum;
   uint16_t mod_p=SABER_P-1;
   uint32_t tid = threadIdx.x, bidx1 = blockIdx.x * SABER_N*SABER_K;
   uint32_t bidx2 = blockIdx.x * SABER_N;
   __shared__ int16_t s0[SABER_K*SABER_N], a[SABER_N];
   for (j = 0; j < SABER_K; j++)
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
       r[bidx2 + tid] = r[bidx2 + tid] & mod_p;   
   }
  
 }

///////////////////////////////////////////////////////////////

__global__ void post_process3(uint16_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x * SABER_N; 

    //for (i = 0; i < SABER_K; i++)
    //{
    in[bid + tid] = MODP(in[bid + tid] + h1);
    //}
}

///////////////////////////////////////////////////////////////

__global__ void msg_unpack_encode_gpu(uint8_t *bytes, uint16_t *data)
{
    size_t i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N, bid2 = blockIdx.x * 64;

    for (i = 0; i < 8; i++)
    {
        data[bid + tid * 8 + i] = ((bytes[bid2 + tid] >> i) & 0x01);
    }

}

///////////////////////////////////////////////////////////////

__global__ void msg_unpack_post_encode_gpu(uint16_t *data)
{
   
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N;

    //for(i=0; i<SABER_N; i++)
    //{
        data[bid + tid] = (data[bid + tid]<<(SABER_EP-1));        
    //}
}

///////////////////////////////////////////////////////////////

__global__ void post_process4(uint16_t *out, uint16_t *in)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N; 
   uint16_t mod_p=SABER_P-1;

   out[bid + tid] = ((out[bid + tid] - in[bid + tid]) & (mod_p)) >> (SABER_EP - SABER_ET);
}

///////////////////////////////////////////////////////////////

__global__ void SABER_pack_5bit(uint8_t *bytes, uint16_t *data){

    uint32_t offset_data=0,offset_byte=0;
    
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N, bid2 =blockIdx.x*SABER_SCALEBYTES_KEM ;
   
    offset_byte= bid2 + 5*tid;
    offset_data= bid + 8*tid;

    bytes[offset_byte + 0]= (data[offset_data + 0] & 0x1f) | ( (data[offset_data + 1] & 0x07)<<5 );
    bytes[offset_byte + 1]= ((data[offset_data + 1] >> 3 ) & 0x03)  | ( (data[offset_data + 2] & 0x1f)<<2 ) | ( (data[offset_data + 3] & 0x01)<<7 );
    bytes[offset_byte + 2]= ((data[offset_data + 3] >> 1 ) & 0x0f)  | ( (data[offset_data + 4] & 0x0f)<<4 );
    bytes[offset_byte + 3]= ((data[offset_data + 4] >> 4 ) & 0x01)  | ( (data[offset_data + 5] & 0x1f)<<1 ) | ( (data[offset_data + 6] & 0x03)<<6 );
    bytes[offset_byte + 4]= ((data[offset_data + 6] >> 2 ) & 0x07)  | ( (data[offset_data + 7] & 0x1f)<<3 );
    }

///////////////////////////////////////////////////////////////

__global__ void post_process5(uint8_t *out, uint8_t *in)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x*SABER_SCALEBYTES_KEM; 

   out[bid + tid] = in[bid2 + tid];
}

///////////////////////////////////////////////////////////////

__device__ void BS2POLp_d_gpu(const uint8_t bytes[SABER_POLYBYTES+SABER_SECRETKEYBYTES], uint16_t data[SABER_N], uint32_t stride)
{
    size_t offset_byte, offset_data;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*stride, bid2 = blockIdx.x*SABER_N*SABER_K;
    offset_byte = 1 * tid + bid;
    offset_data = 4 * tid + bid2;
    
    data[offset_data]   = ( (bytes[offset_byte] & 0x03) ^ 0x2 ) - 0x2;
    data[offset_data+1] = ( ((bytes[offset_byte]>>2) & 0x03) ^ 0x2 ) - 0x2;
    data[offset_data+2] = ( ((bytes[offset_byte]>>4) & 0x03) ^ 0x2 ) - 0x2;
    data[offset_data+3] = ( ((bytes[offset_byte]>>6) & 0x03) ^ 0x2 ) - 0x2;
}

///////////////////////////////////////////////////////////////

__global__ void BS2POLVECp_d_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride)
{
    size_t i;
    for (i = 0; i < SABER_K; i++)
    {
        BS2POLp_d_gpu(bytes + i * (2 * SABER_N / 8), data + i*SABER_N, stride);
    }
}

//////////////////////////////////////////////////////////////



 //////////////////////////////////////////////////////////////

 __global__ void post_process6(uint8_t *out, uint8_t *in)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x*SABER_SCALEBYTES_KEM; 

   out[bid2 + tid] = in[bid + tid];
}

///////////////////////////////////////////////////////////////

__global__ void SABER_un_pack5bit(uint8_t *bytes, uint16_t *data)
{
    
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_SCALEBYTES_KEM, bid2 = blockIdx.x * SABER_N; 

    uint32_t offset_byte = bid + 5*tid;
    uint32_t offset_data = bid2 + 8*tid;      
    
    data[offset_data + 0] = (bytes[offset_byte+0])&0x1f;
    data[offset_data + 1] = ( ( (bytes[offset_byte+0])>>5 )&0x07) | ( ( (bytes[offset_byte+1])&0x03)<<3 );
    data[offset_data + 2] = ( ( (bytes[offset_byte+1])>>2 )&0x1f);
    data[offset_data + 3] = ( ( (bytes[offset_byte+1])>>7 )&0x01) | ( ( (bytes[offset_byte+2])&0x0f)<<1 );
    data[offset_data + 4] = ( ( (bytes[offset_byte+2])>>4 )&0x0f) | ( ( (bytes[offset_byte+3])&0x01)<<4 );
    data[offset_data + 5] = ( ( (bytes[offset_byte+3])>>1 )&0x1f);
    data[offset_data + 6] = ( ( (bytes[offset_byte+3])>>6 )&0x03) | ( ( (bytes[offset_byte+4])&0x07)<<2 );
    data[offset_data + 7] = ( (bytes[offset_byte+4]>>3)&0x1f );

}

///////////////////////////////////////////////////////////////

__global__ void post_processnull(uint16_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x * SABER_N; 
    
      in[bid + tid] = 0;
    
}

///////////////////////////////////////////////////////////////

__global__ void post_process7(uint16_t *out, uint16_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N; 
    //uint16_t mod_p=SABER_P-1;

    out[bid + tid] = MODP( out[bid + tid] + h2 - (in[bid + tid]<<(SABER_EP-SABER_ET)) )  >> (SABER_EP-1);
}

///////////////////////////////////////////////////////////////

__global__ void POL2MSG(uint16_t *message_dec_unpacked, uint8_t *message_dec)
{

    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N, bid2 = blockIdx.x*64; 
    int32_t i;


        message_dec[bid2 + tid] = 0;
        for(i=0; i<8; i++)
        {
        message_dec[bid2 + tid] = message_dec[bid2 + tid] | (message_dec_unpacked[bid + tid*8 + i] <<i);
        }
}

///////////////////////////////////////////////////////////////

__global__ void post_process8(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x*64, bid2 = blockIdx.x*SABER_SECRETKEYBYTES; 

    out[bid + tid] = in[bid2 + tid];
}

///////////////////////////////////////////////////////////////

__global__ void verify_gpu(uint64_t *r, uint8_t *a, uint8_t *b, size_t len)
{
    // uint64_t r;
    size_t i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC;   
    
    r[tid] = 0;
    for (i = 0; i < len/blockDim.x; i++)
    {
        r[bid + i*blockDim.x + tid]|= a[bid + i*blockDim.x + tid] ^ b[bid + i*blockDim.x + tid];
        r[bid + i*blockDim.x + tid] = (-r[bid + i*blockDim.x + tid]) >> 63;
        // if(r[bid + i*blockDim.x + tid])
        // { 
        //  printf("Not same %u %u!\n", i, tid);        
        // }
    }
}

///////////////////////////////////////////////////////////////

__global__ void cmov_gpu(uint8_t *r, uint8_t *x, size_t len, uint64_t *b)
{
    size_t i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x*64, bid2 = blockIdx.x*SABER_SECRETKEYBYTES, bid3 = blockIdx.x*SABER_BYTES_CCA_DEC ;        
    for (i = 0; i < len/blockDim.x; i++)
    {
        b[bid3 + i*blockDim.x + tid] = -b[bid3 + i*blockDim.x + tid];
  // for (i = 0; i < len; i++)
        r[bid + i*blockDim.x + tid] ^= b[bid3 + i*blockDim.x + tid] & (x[bid2 + i*blockDim.x + tid] ^ r[bid + i*blockDim.x + tid]);
    }
}

////////////////////////////////////////////////////////

__device__ int16_t reduce(int16_t a, int64_t p)
{
    return a&(p-1);
}

////////////////////////////////////////////////////////

__device__ void karatsuba_simple(const uint16_t* a_1,const uint16_t* b_1, uint16_t* result_final){//uses 10 registers
    uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    const uint16_t N=64;
    uint16_t d01[N/2-1];
    uint16_t d0123[N/2-1];
    uint16_t d23[N/2-1];
    uint16_t result_d01[N-1];   

    int32_t i,j;

    memset(result_d01,0,(N-1)*sizeof(uint16_t));
    memset(d01,0,(N/2-1)*sizeof(uint16_t));
    memset(d0123,0,(N/2-1)*sizeof(uint16_t));
    memset(d23,0,(N/2-1)*sizeof(uint16_t));
    memset(result_final,0,(2*N-1)*sizeof(uint16_t));

    uint16_t acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,acc10;


    for (i = idx; i < N/4; i++) {
        acc1=a_1[i];//a0
        acc2=a_1[i+N/4];//a1
        acc3=a_1[i+2*N/4];//a2
        acc4=a_1[i+3*N/4];//a3  
        for (j = 0; j < N/4; j++) {

            acc5=b_1[j];//b0
            acc6=b_1[j+N/4];//b1

            result_final[i+j+0*N/4]=result_final[i+j+0*N/4]+acc1*acc5;
            result_final[i+j+2*N/4]=result_final[i+j+2*N/4]+acc2*acc6;

            acc7=acc5+acc6;//b01
            acc8=acc1+acc2;//a01
            d01[i+j]=d01[i+j] + acc7*acc8;
    //--------------------------------------------------------

            acc7=b_1[j+2*N/4];//b2
            acc8=b_1[j+3*N/4];//b3          
            result_final[i+j+4*N/4]=result_final[i+j+4*N/4]+acc7*acc3;

            result_final[i+j+6*N/4]=result_final[i+j+6*N/4]+acc8*acc4;

            acc9=acc3+acc4;
            acc10=acc7+acc8;
            d23[i+j]=d23[i+j] + acc9*acc10;
    //--------------------------------------------------------

            acc5=acc5+acc7;//b02
            acc7=acc1+acc3;//a02
            result_d01[i+j+0*N/4]=result_d01[i+j+0*N/4]+acc5*acc7;

            acc6=acc6+acc8;//b13
            acc8=acc2+acc4;         
            result_d01[i+j+ 2*N/4]=result_d01[i+j+ 2*N/4]+acc6*acc8;

            acc5=acc5+acc6;
            acc7=acc7+acc8;
            d0123[i+j]=d0123[i+j] + acc5*acc7;
        }
    }

//------------------2nd last stage-------------------------

    for(i=idx;i<N/2-1;i++){
        d0123[i]=d0123[i]-result_d01[i+0*N/4]-result_d01[i+2*N/4];
        d01[i]=d01[i]-result_final[i+0*N/4]-result_final[i+2*N/4];
        d23[i]=d23[i]-result_final[i+4*N/4]-result_final[i+6*N/4];
    }

    for(i=idx;i<N/2-1;i++){
        result_d01[i+1*N/4]=result_d01[i+1*N/4]+d0123[i];
        result_final[i+1*N/4]=result_final[i+1*N/4]+d01[i];
        result_final[i+5*N/4]=result_final[i+5*N/4]+d23[i];
    }

//------------Last stage---------------------------
    for(i=idx;i<N-1;i++){
        result_d01[i]=result_d01[i]-result_final[i]-result_final[i+N];
    }
    
    for(i=idx;i<N-1;i++){
        result_final[i+1*N/2]=result_final[i+1*N/2]+result_d01[i];//-result_d0[i]-result_d1[i];     
    }

}

///////////////////////////////////////////////////////

__device__ void toom_cook_4way (const uint16_t* a1,const uint16_t* b1, uint16_t* result)
{
    uint16_t inv3 = 43691, inv9 = 36409, inv15 = 61167;
    uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint16_t aw1[N_SB], aw2[N_SB], aw3[N_SB], aw4[N_SB], aw5[N_SB], aw6[N_SB], aw7[N_SB];
    uint16_t bw1[N_SB], bw2[N_SB], bw3[N_SB], bw4[N_SB], bw5[N_SB], bw6[N_SB], bw7[N_SB];
    uint16_t w1[N_SB_RES] = {0}, w2[N_SB_RES] = {0}, w3[N_SB_RES] = {0}, w4[N_SB_RES] = {0},
             w5[N_SB_RES] = {0}, w6[N_SB_RES] = {0}, w7[N_SB_RES] = {0};
    uint16_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint16_t *A0, *A1, *A2, *A3, *B0, *B1, *B2, *B3;
    A0 = (uint16_t*)a1;
    A1 = (uint16_t*)&a1[N_SB];
    A2 = (uint16_t*)&a1[2*N_SB];
    A3 = (uint16_t*)&a1[3*N_SB];
    B0 = (uint16_t*)b1;
    B1 = (uint16_t*)&b1[N_SB];
    B2 = (uint16_t*)&b1[2*N_SB];
    B3 = (uint16_t*)&b1[3*N_SB];

    uint16_t * C;
    C = result;

    int i,j;

// EVALUATION
    for (j = idx; j < N_SB; ++j) {
        r0 = A0[j];
        r1 = A1[j];
        r2 = A2[j];
        r3 = A3[j];
        r4 = r0 + r2;
        r5 = r1 + r3;
        r6 = r4 + r5; r7 = r4 - r5;
        aw3[j] = r6;
        aw4[j] = r7;
        r4 = ((r0 << 2)+r2) << 1;
        r5 = (r1 << 2) + r3;
        r6 = r4 + r5; r7 = r4 - r5;
        aw5[j] = r6;
        aw6[j] = r7;
        r4 = (r3 << 3) + (r2 << 2) + (r1 << 1) + r0;
        aw2[j] = r4; aw7[j] = r0;
        aw1[j] = r3;
    }
    for (j = idx; j < N_SB; ++j) {
        r0 = B0[j];
        r1 = B1[j];
        r2 = B2[j];
        r3 = B3[j];
        r4 = r0 + r2;
        r5 = r1 + r3;
        r6 = r4 + r5; r7 = r4 - r5;
        bw3[j] = r6;
        bw4[j] = r7;
        r4 = ((r0 << 2)+r2) << 1;
        r5 = (r1 << 2) + r3;
        r6 = r4 + r5; r7 = r4 - r5;
        bw5[j] = r6;
        bw6[j] = r7;
        r4 = (r3 << 3) + (r2 << 2) + (r1 << 1) + r0;
        bw2[j] = r4; bw7[j] = r0;
        bw1[j] = r3;
    }

// MULTIPLICATION

    karatsuba_simple(aw1, bw1, w1);
    karatsuba_simple(aw2, bw2, w2);
    karatsuba_simple(aw3, bw3, w3);
    karatsuba_simple(aw4, bw4, w4);
    karatsuba_simple(aw5, bw5, w5);
    karatsuba_simple(aw6, bw6, w6);
    karatsuba_simple(aw7, bw7, w7);

// INTERPOLATION
    for (i = idx; i < N_SB_RES; ++i) {
        r0 = w1[i];
        r1 = w2[i];
        r2 = w3[i];
        r3 = w4[i];
        r4 = w5[i];
        r5 = w6[i];
        r6 = w7[i];

        r1 = r1 + r4;
        r5 = r5 - r4;
        r3 = ((r3-r2) >> 1);
        r4 = r4 - r0;
        r4 = r4 - (r6 << 6);
        r4 = (r4 << 1) + r5;
        r2 = r2 + r3;
        r1 = r1 - (r2 << 6) - r2;
        r2 = r2 - r6;
        r2 = r2 - r0;
        r1 = r1 + 45*r2;
        r4 = (((r4 - (r2 << 3))*inv3) >> 3);
        r5 = r5 + r1;
        r1 = (((r1 + (r3 << 4))*inv9) >> 1);
        r3 = -(r3 + r1);
        r5 = (((30*r1 - r5)*inv15) >> 2);
        r2 = r2 - r4;
        r1 = r1 - r5;

        C[i]     += r6;
        C[i+64]  += r5;
        C[i+128] += r4;
        C[i+192] += r3;
        C[i+256] += r2;
        C[i+320] += r1;
        C[i+384] += r0;
    }
}

////////////////////////////////////////////////////////

__device__ void pol_mul(uint16_t* a, uint16_t* b, uint16_t* res, uint16_t p, uint32_t n)
{ 
   
    uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t i;

//-------------------normal multiplication-----------------

    uint16_t c[2*SABER_N];
    memset(c,0,(2*n)*sizeof(uint16_t));

    toom_cook_4way(a, b, c);
    
    for(i=0;i<n;i++)
        res[i] = reduce(c[i]-c[i+n],p);
    
}

////////////////////////////////////////////////////////

__global__ void InnerProd(uint16_t* pkcl, uint16_t* skpv, uint16_t mod, uint16_t* res)
{


    uint32_t j,k;
    uint16_t acc[SABER_N]; 

    // vector-vector scalar multiplication with mod p
    for(j=0;j<SABER_K;j++){
        pol_mul(pkcl+(j*SABER_N), skpv+(j*SABER_N), acc , SABER_P, SABER_N);

            for(k=0;k<SABER_N;k++){
                res[k]=res[k]+acc[k];
                res[k]=res[k]& mod; //reduction
                acc[k]=0; //clear the accumulator
        }
    }
}

