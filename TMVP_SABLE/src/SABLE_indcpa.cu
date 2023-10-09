#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <curand.h>
#include <cublas_v2.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../include/SABLE_indcpa.cuh"
#include "../include/SABLE_params.h"
#include "../include/pack.cuh"
#include "../include/fips202.cuh"
#include "../include/poly.cuh"
#include "../include/tv.h"

////////////////////////////////////////////////////////////// 
    
void SABLE_enc(uint8_t *h_k, uint8_t *h_pk, uint8_t *h_c, int mode)
{
   
                        //Encryption//
    uint8_t *testprint;
    cudaMallocHost((void**) &testprint, BATCH*64 * sizeof(uint8_t));

    int i, j;
    uint16_t *d_a, *host_fp16, *p1, *p2, *p3, *arr, *rearr;
    uint8_t *d_pk, *d_c, *d_m, *d_kr, *d_A8, *d_buf, *d_k; 
    uint16_t *pkcl; 
    uint16_t *skpv1;
    uint16_t *message;
    uint16_t *res;   
    uint16_t *vprime;
    uint8_t *msk_c, *gen_buf;
    half *h_fp16, *r_fp16, *h_da, *h_skpv1;
    half *a1, *b1, *b1_m, *a2, *b2, *b2_m, *a3, *b3, *b3_m;
    float *c_wmma2, *host_float, *c_wmma1, *c_wmma3, *c_wmma4;
    uint8_t *h_m, *h_m1;
    uint16_t *ac1, *bc1, *ac2, *bc2, *ac3, *bc3;

    cudaEvent_t start, stop, startIP, startMV, stopIP, stopMV;
    float elapsed, elapsedIP;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventCreate(&startIP);    cudaEventCreate(&stopIP);
    cudaEventCreate(&startMV);    cudaEventCreate(&stopMV);

    
    cudaMalloc((void**) &h_skpv1, SABER_N * SABER_N * sizeof(half));
    cudaMallocHost((void**) &h_m, BATCH*64* sizeof(uint8_t));
    cudaMallocHost((void**) &host_fp16, BATCH*SABER_N*SABER_N * sizeof(uint16_t));
    cudaMallocHost((void**) &host_float, SABER_N*SABER_N * sizeof(float));
    cudaMallocHost((void**) &h_m1, BATCH*SABER_KEYBYTES* sizeof(uint8_t));

    cudaMalloc((void**) &a1, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b1, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a2, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b2, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a3, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b3, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &b1_m, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &b2_m, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b3_m, SABER_N/2 * SABER_N/2 * sizeof(half));

    cudaMalloc((void**) &ac1, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc1, BATCH*(SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));
    cudaMalloc((void**) &ac2, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc2, BATCH*(SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));
    cudaMalloc((void**) &ac3, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc3, BATCH*(SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));
 
    cudaMalloc((void**) &h_da, SABER_N * SABER_N * sizeof(half));   
    cudaMalloc((void**) &c_wmma1, SABER_N/2*SABER_N/2 * sizeof(float));   
    cudaMalloc((void**) &c_wmma2, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma3, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma4, SABER_N * SABER_N * sizeof(float));
    cudaMalloc((void**) &h_fp16, SABER_N * SABER_N * sizeof(half));   
    cudaMalloc((void**) &r_fp16, SABER_N * SABER_N * sizeof(half));
    cudaMalloc((void**) &d_a, BATCH*SABER_K*SABER_N*SABER_K * sizeof(uint16_t));
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_pk, BATCH*SABER_PUBLICKEYBYTES* sizeof(uint8_t));     
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));     
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES * sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t));
    cudaMalloc((void**) &vprime, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &p1, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p2, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p3, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &arr, SABER_N*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &rearr, (SABER_N*3*SABER_N)/4* sizeof(uint16_t));
    cudaMalloc((void**) &res, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &message, BATCH*SABER_KEYBYTES*8* sizeof(uint16_t));
    cudaMalloc((void**) &skpv1, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &pkcl, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &msk_c, BATCH*SABER_SCALEBYTES_KEM* sizeof(uint8_t));
    cudaMalloc((void**) &gen_buf, BATCH*buf_size* sizeof(uint8_t));
    cudaMalloc((void**) &d_A8, BATCH*SABER_K*SABER_A8 * sizeof(uint8_t));

    

    for(j=0; j<BATCH; j++) 
        for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) 
            h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk_tv[i];

    for(j=0; j<BATCH; j++) 
        for(i=0; i<64; i++) 
            h_m[j*64 + i] = buf_tv[i];

    for(j=0; j<BATCH; j++) 
        for(i=0; i<SABER_KEYBYTES; i++) 
            h_m1[j*SABER_KEYBYTES + i] = m_tv[i];     

    uint32_t threads = 32 * ((SABER_N/2)/WMMA_M)*((SABER_N/2)/WMMA_M);// each warp computes 16x16 matrix
    uint32_t blocks = 1;
    if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }

    
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_kr, h_k, BATCH*64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, h_m, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m1, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
cudaEventRecord(start);

///////////////////////////////////////////////////////////////
                     /*Encalsulation*/
///////////////////////////////////////////////////////////////

sha3_256_gpu<<<1,BATCH>>>(d_buf, d_buf, 32, 64, 64);
sha3_256_gpu<<<1,BATCH>>>(d_buf + 32, d_pk, SABER_INDCPA_PUBLICKEYBYTES, SABER_INDCPA_PUBLICKEYBYTES, 64);  
sha3_512_gpu<<<1,BATCH>>>(d_kr, d_buf, 64);

///////////////////////////////////////////////////////////////
                        /*Encryption*/

cudaEventRecord(startIP);
///////////////////////////////////////////////////////////////
                        /*GenMatrix*/
shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_K*SABER_A8, SABER_K*SABER_A8);
BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_a);  

///////////////////////////////////////////////////////////////
                        /*GenSecret*/
/*shake128_gpu<<<BATCH, 32>>>(gen_buf, d_kr + 32, SABER_NOISESEEDBYTES, buf_size, buf_size);
    for(i=0;i<SABER_K;i++){
cbd<<<BATCH, SABER_N/4>>>(skpv1 + i*SABER_N, gen_buf+i*SABER_MU*SABER_N/8);
    }
*/
GenSecret_gpu<<<1,BATCH>>>(skpv1, d_kr + 32);

///////////////////////////////////////////////////////////////
if (mode == 0)
{
    for (int j = 0; j < SABER_K; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, skpv1 + j*SABER_N);
        submatrix_m2 <<< SABER_N/2, SABER_N/2 >>>(ac1, ac2, ac3, arr);
    for (int i = 0; i < SABER_K; i++){
        submatrix_m1 <<< BATCH, SABER_N/2 >>>(bc1, bc2, bc3, d_a + i*SABER_N*SABER_K + j*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>> (ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cudaq <<< BATCH, SABER_N/2 >>>(p1, p2, p3, res + i*SABER_N);
        }}
}
else if(mode == 1)
 {
    for (int j = 0; j < SABER_K; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, skpv1 + j*SABER_N);
        submatrix_m2 <<< SABER_N/2, SABER_N/2 >>>(ac1, ac2, ac3, arr);
    for (int i = 0; i < SABER_K; i++){
        submatrix_m1 <<< BATCH, SABER_N/2 >>>(bc1, bc2, bc3, d_a + i*SABER_N*SABER_K + j*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>> (ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cudaq <<< BATCH, SABER_N/2 >>>(p1, p2, p3, res + i*SABER_N);
        }}
 }
else
{
        MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(res, d_a, skpv1);
}

post_process<<<BATCH, SABER_N>>>(res);
POLVECp2BS_gpu<<<BATCH, SABER_N / 8>>>(d_c, res);
BS2POLVECp_gpu<<<BATCH, SABER_N / 8>>>(d_pk, pkcl, SABER_INDCPA_PUBLICKEYBYTES);

post_processnull<<<BATCH,SABER_N>>>(vprime);
post_process2<<<BATCH, SABER_N>>>(skpv1);
if (mode == 0)
{
    for (int i = 0; i < SABER_K; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, skpv1 + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(a1, b1, a2, b2, a3, b3, arr, pkcl + i*SABER_N); 
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(vprime, c_wmma1, c_wmma2, c_wmma3); 
    }
}
else if (mode == 1)
{
    for (int i = 0; i < SABER_K; i++)
    {
    convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, skpv1 + i*SABER_N);
    submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(ac1, bc1, ac2, bc2, ac3, bc3, arr, pkcl + i*SABER_N);
    matvecp_cuda <<< BATCH, SABER_N >>>(ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
    matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, vprime);
    }
}
else
{
    VecVecMul_Inner_gpu<<<BATCH, SABER_N>>>(vprime, pkcl, skpv1);
}
post_process3<<<BATCH, SABER_N>>>(vprime);
msg_unpack_encode_gpu<<<BATCH, SABER_KEYBYTES>>>(d_buf, message);
msg_unpack_post_encode_gpu<<<BATCH, SABER_N>>>(message);

post_process4<<<BATCH, SABER_N>>>(vprime, message);
SABER_pack_5bit<<<BATCH, SABER_N/8>>>(msk_c, vprime);

post_process5<<<BATCH, SABER_SCALEBYTES_KEM>>>(d_c+SABER_POLYVECCOMPRESSEDBYTES, msk_c);

cudaEventRecord(stopIP);
///////////////////////////////////////////////////////////////

sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64); 
sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, 32); 

///////////////////////////////////////////////////////////////

cudaMemcpy(h_c, d_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyDeviceToHost); 
cudaMemcpy(h_k, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventSynchronize(stopIP);    
cudaEventElapsedTime(&elapsed, start, stop);
cudaEventElapsedTime(&elapsedIP, startIP, stopIP);


cudaError_t cudaerr = cudaDeviceSynchronize();
if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error %s.",cudaGetErrorString(cudaerr));

printf("\n");
printf("encrypt: gpu u16 mode took %fms, Throughput %f Encaps/s\n and Throughput %f Encryption/s\n", elapsed, BATCH*1000/elapsed, BATCH*1000/elapsedIP);

cudaMemcpy(testprint, d, 4*4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
cudaMemcpy(host_fp16, res, SABER_N*SABER_K * sizeof(uint16_t), cudaMemcpyDeviceToHost);
cudaMemcpy(host_float, c_wmma4, 256*256 * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_skpv1, b1_m, 128*128 * sizeof(half), cudaMemcpyDeviceToHost);
/*
printf("\n h_ka:\n"); 
for(j=0; j<BATCH; j++) {
    printf("\nbatch: %u\n", j); 
    for(i=0; i<SABER_KEYBYTES; i++) 
        printf("%u ", h_k[j*SABER_KEYBYTES + i]);}
printf("\n");


    printf("h_res=\n");
    for(i=0; i<BATCH*SABER_N*SABER_K; i++)
    printf("0x%u, ", host_fp16[i]);
    printf("\n");
*/

cudaFree(pkcl);     cudaFree(d_kr);
cudaFree(skpv1);    cudaFree(msk_c);
cudaFree(res);      cudaFree(d_A8);
cudaFree(d_pk);     cudaFree(d_a);
cudaFree(d_c);      cudaFree(message);
cudaFree(res);      cudaFree(skpv1);
cudaFree(pkcl);     cudaFree(d_k);
cudaFree(gen_buf);  cudaFree(h_fp16);
cudaFree(r_fp16);   cudaFree(c_wmma2);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SABLE_dec(uint8_t *h_c, uint8_t *h_pk, uint8_t *h_sk, uint8_t *h_k, int mode)
{
                        //Decryption//
    uint8_t *testprint;
    cudaMallocHost((void**) &testprint, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t));

    uint32_t i,j;
    uint16_t *sksv, *float_fp16, *skpv1, *res, *pkcl, *vprime, *arr, *rearr, *d_arr, *d_rearr; 
    uint16_t *pksv, *d_a, *message, *p1, *p2, *p3, *d_p1, *d_p2, *d_p3;
    uint8_t *scale_ar, *gen_buf, *msk_c, *h_km;
    uint16_t *v, *h_s, *h_b;
    uint16_t *op;
    uint64_t *d_r;
    uint8_t *d_sk, *d_c, *d_m, *d_message, *d_buf, *d_pk, *d_cCompare, *d_k, *d_A8, *d_kr;
    half *h_fp16, *s_fp16, *h_da, *h_skpv1;
    float *c_wmma4;

    half *a1, *b1, *a2, *b2, *a3, *b3;
    float *c_wmma2, *c_wmma1, *c_wmma3;

    half *d_a1, *d_b1, *d_a2, *d_b2, *d_a3, *d_b3;
    float *d_wmma2, *d_wmma1, *d_wmma3;

    uint16_t *ac1, *bc1, *ac2, *bc2, *ac3, *bc3;
    uint16_t *d_ac1, *d_bc1, *d_ac2, *d_bc2, *d_ac3, *d_bc3;
    

    cudaMallocHost((void**) &float_fp16, BATCH*SABER_N * sizeof(uint16_t));
    cudaMallocHost((void**) &h_km, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMallocHost((void**) &h_s, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMallocHost((void**) &h_b, BATCH*SABER_K*SABER_N* sizeof(uint16_t));

    cudaMalloc((void**) &h_da, SABER_N * SABER_N * sizeof(half));   
    cudaMalloc((void**) &h_skpv1, SABER_N * SABER_N * sizeof(half));
    cudaMalloc((void**) &h_fp16, SABER_N * SABER_N * sizeof(half));   
    cudaMalloc((void**) &s_fp16, SABER_N * SABER_N * sizeof(half));
    cudaMalloc((void**) &d_r, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint64_t));
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_A8, BATCH*SABER_K*SABER_A8 * sizeof(uint8_t));
    cudaMalloc((void**) &d_a, BATCH*SABER_K*SABER_N*SABER_K * sizeof(uint16_t));       
    cudaMalloc((void**) &d_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES* sizeof(uint8_t));   
    cudaMalloc((void**) &d_sk, BATCH*SABER_SECRETKEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));
    cudaMalloc((void**) &d_cCompare, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));    
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));       
    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_message, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &skpv1, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &gen_buf, BATCH*buf_size* sizeof(uint8_t));
    cudaMalloc((void**) &res, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &pkcl, BATCH*SABER_K*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &vprime, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &message, BATCH*SABER_KEYBYTES*8* sizeof(uint16_t));
    cudaMalloc((void**) &msk_c, BATCH*SABER_SCALEBYTES_KEM* sizeof(uint8_t));
    
    cudaMalloc((void**) &p1, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p2, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p3, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &arr, SABER_N*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &rearr, (SABER_N*3*SABER_N)/4* sizeof(uint16_t));
    cudaMalloc((void**) &d_p1, BATCH*(SABER_N/2)* sizeof(uint16_t));
    cudaMalloc((void**) &d_p2, BATCH*(SABER_N/2)* sizeof(uint16_t));
    cudaMalloc((void**) &d_p3, BATCH*(SABER_N/2)* sizeof(uint16_t));
    cudaMalloc((void**) &d_arr, SABER_N*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_rearr, (SABER_N*3*SABER_N)/4* sizeof(uint16_t));
 
    cudaMalloc((void**) &sksv, BATCH*SABER_K*SABER_N * sizeof(uint16_t));
    cudaMalloc((void**) &pksv, BATCH*SABER_K*SABER_N * sizeof(uint16_t));
    cudaMalloc((void**) &scale_ar, BATCH*SABER_SCALEBYTES_KEM * sizeof(uint8_t));
    cudaMalloc((void**) &v, BATCH*SABER_N * sizeof(uint16_t));
    cudaMalloc((void**) &op, BATCH*SABER_N * sizeof(uint16_t));

    cudaMalloc((void**) &a1, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b1, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a2, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b2, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a3, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b3, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &c_wmma1, SABER_N/2*SABER_N/2 * sizeof(float));   
    cudaMalloc((void**) &c_wmma2, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma3, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma4, SABER_N*SABER_N * sizeof(float));

    cudaMalloc((void**) &d_a1, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &d_b1, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &d_a2, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &d_b2, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &d_a3, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &d_b3, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &d_wmma1, SABER_N/2*SABER_N/2 * sizeof(float));   
    cudaMalloc((void**) &d_wmma2, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &d_wmma3, SABER_N/2*SABER_N/2 * sizeof(float));

    cudaMalloc((void**) &ac1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc1, BATCH*SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &ac2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc2, BATCH*SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &ac3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc3, BATCH*SABER_N/2 * SABER_N/2 * sizeof(uint16_t));

    cudaMalloc((void**) &d_ac1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &d_ac2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &d_ac3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    

    cudaEvent_t start, stop, startIP, startMV, stopIP, stopMV;
    float elapsed, elapsedIP;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventCreate(&startIP);    cudaEventCreate(&stopIP);
    cudaEventCreate(&startMV);    cudaEventCreate(&stopMV);

    for(j=0; j<BATCH; j++) 
        for(i=0; i<SABER_SECRETKEYBYTES; i++) 
            h_sk[j*SABER_SECRETKEYBYTES + i] = sk_tv[i];
     
    for(j=0; j<BATCH; j++) 
        for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) 
            h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk_tv[i];
    
    uint32_t threads = 32 * ((SABER_N/2)/WMMA_M)*((SABER_N/2)/WMMA_M);// each warp computes 16x16 matrix
    uint32_t blocks = 1;
    if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }
    
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_sk, h_sk, BATCH*SABER_SECRETKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(sksv, h_s, BATCH*SABER_K*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pksv, h_b, BATCH*SABER_K*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);

cudaEventRecord(start); 
cudaEventRecord(startIP); 
///////////////////////////////////////////////////////////////
                        /*Decryption*/
BS2POLVECp_d_gpu<<<BATCH, SABER_N/4>>>(d_sk, sksv, SABER_SECRETKEYBYTES);
BS2POLVECp_gpu<<<BATCH,   SABER_N/8>>>(d_c, pksv, SABER_BYTES_CCA_DEC);

post_process2<<<BATCH, SABER_N>>>(sksv);
post_processnull<<<BATCH,SABER_N>>>(v);

if (mode == 0)
{
    for (int i = 0; i < SABER_K; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(d_arr, sksv + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(d_a1, d_b1, d_a2, d_b2, d_a3, d_b3, d_arr, pksv + i*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (d_a1, d_b1, d_a2, d_b2, d_a3, d_b3, d_wmma1, d_wmma2, d_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(v, d_wmma1, d_wmma2, d_wmma3); 
    }
}
else if(mode == 1)
{
    for (int i = 0; i < SABER_K; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, sksv + i*SABER_N);
        submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(ac1, bc1, ac2, bc2, ac3, bc3, arr, pksv + i*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>>(ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, v);
    }
}
else
{
    VecVecMul_Inner_gpu<<<BATCH, SABER_N>>>(v, pksv, sksv);
}
post_process6<<<BATCH, SABER_SCALEBYTES_KEM>>>(scale_ar, d_c + SABER_POLYVECCOMPRESSEDBYTES);

SABER_un_pack5bit<<<BATCH, SABER_N/8>>>(scale_ar, op);
post_process7<<<BATCH, SABER_N>>>(v, op);

POL2MSG<<< BATCH, SABER_KEYBYTES>>>(v, d_message);
cudaEventRecord(stopIP);
///////////////////////////////////////////////////////////////
                        /*Decapsulation*/
post_process8<<<BATCH, SABER_KEYBYTES>>>(d_message + 32, d_sk + SABER_SECRETKEYBYTES - 64);

sha3_512_gpu<<<1,BATCH>>>(d_kr, d_message, 64);

//////////////////////////////////////////////////////////////
                        /*Indcpa_Kem_Enc*/

                        /*GenMatrix*/
shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_K*SABER_A8, SABER_K*SABER_A8);
BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_a);  

                        /*GenSecret*/

GenSecret_gpu<<<1,BATCH>>>(skpv1, d_kr + 32);
if (mode == 0)
{
    Mul_process <<< BATCH, SABER_N*SABER_K >>> (skpv1);
    for (int j = 0; j < SABER_K; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, skpv1 + j*SABER_N);
        submatrix_m1_tensor <<< SABER_N/2, SABER_N/2 >>>(a1, a2, a3, arr);
    for (int i = 0; i < SABER_K; i++){
        submatrix_m2_tensor <<< SABER_N/2, SABER_N/2 >>>(b1, b2, b3, d_a + i*SABER_N*SABER_K + j*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP_m<<< BATCH, SABER_N/2 >>>(res + i*SABER_N, c_wmma1, c_wmma2, c_wmma3);
    }}
}
else if (mode == 1)
{
    for (int j = 0; j < SABER_K; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, skpv1 + j*SABER_N);
        submatrix_m2 <<< SABER_N/2, SABER_N/2 >>>(ac1, ac2, ac3, arr);
    for (int i = 0; i < SABER_K; i++){
        submatrix_m1 <<< BATCH, SABER_N/2 >>>(bc1, bc2, bc3, d_a + i*SABER_N*SABER_K + j*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>> (ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cudaq <<< BATCH, SABER_N/2 >>>(p1, p2, p3, res + i*SABER_N);
    }}
}
else
{
    MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(res, d_a, skpv1); 
}

post_process<<<BATCH, SABER_N>>>(res);
POLVECp2BS_gpu<<<BATCH, SABER_N / 8>>>(d_cCompare, res);
BS2POLVECp_gpu<<<BATCH, SABER_N / 8>>>(d_pk, pkcl, SABER_INDCPA_PUBLICKEYBYTES);

post_processnull<<<BATCH,SABER_N>>>(vprime);
post_process2<<<BATCH, SABER_N>>>(skpv1);

if (mode == 0)
{
    for (int i = 0; i < SABER_K; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, skpv1 + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(a1, b1, a2, b2, a3, b3, arr, pkcl + i*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(vprime, c_wmma1, c_wmma2, c_wmma3); 
    }
}
else if (mode == 1)
{
    for (int i = 0; i < SABER_K; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, skpv1 + i*SABER_N);
        submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(d_ac1, d_bc1, d_ac2, d_bc2, d_ac3, d_bc3, arr, pkcl + i*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>>(d_ac1, d_bc1, d_ac2, d_bc2, d_ac3, d_bc3, d_p1, d_p2, d_p3);
        matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, vprime);
    }
}
else
{
    VecVecMul_Inner_gpu<<<BATCH, SABER_N>>>(vprime, pkcl, skpv1);
}

post_process3<<<BATCH, SABER_N>>>(vprime);
msg_unpack_encode_gpu<<<BATCH, SABER_KEYBYTES>>>(d_buf, message);
msg_unpack_post_encode_gpu<<<BATCH, SABER_N>>>(message);

post_process4<<<BATCH, SABER_N>>>(vprime, message);
SABER_pack_5bit<<<BATCH, SABER_N/8>>>(msk_c, vprime);

post_process5<<<BATCH, SABER_SCALEBYTES_KEM>>>(d_cCompare+SABER_POLYVECCOMPRESSEDBYTES, msk_c);

// ************* end of indcpa_kem_enc *************

verify_gpu<<<BATCH, SABER_N>>>(d_r, d_c, d_cCompare, SABER_BYTES_CCA_DEC);

sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64);
sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, SABER_KEYBYTES);

cmov_gpu<<<BATCH,SABER_N>>>(d_kr, d_sk + SABER_SECRETKEYBYTES - SABER_KEYBYTES, SABER_KEYBYTES, d_r);



///////////////////////////////////////////////////////////////

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventSynchronize(stopIP);    
cudaEventElapsedTime(&elapsed, start, stop);
cudaEventElapsedTime(&elapsedIP, startIP, stopIP);

cudaError_t cudaerr = cudaDeviceSynchronize();
if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error %s.",cudaGetErrorString(cudaerr));

printf("\n");
printf("decrypt: gpu u16 mode took %fms, Throughput %f Decap/s\n and Throughput %f Decryption/s\n", elapsed, BATCH*1000/elapsed, BATCH*1000/elapsedIP);

 cudaMemcpy(h_km, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);

 cudaMemcpy(float_fp16, v, BATCH*SABER_N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
 cudaMemcpy(testprint, scale_ar, BATCH*SABER_SCALEBYTES_KEM * sizeof(uint8_t), cudaMemcpyDeviceToHost);
/*
  printf("h_kb=\n");
    for(i=0; i<BATCH*SABER_N; i++)
    printf("0x%u, ", float_fp16[i]);
    printf("\n");

printf("\n h_kb:\n"); 
for(j=0; j<BATCH; j++) {
    printf("\nbatch: %u\n", j); 
    for(i=0; i<SABER_KEYBYTES; i++) 
        printf("%u ", h_km[j*SABER_KEYBYTES + i]);}
printf("\n");
*/
 for(j=0; j<BATCH; j++)
    {
        for(i=0; i<SABER_KEYBYTES; i++)
        {
            if(h_km[j*SABER_KEYBYTES + i]!=h_k[j*SABER_KEYBYTES + i]){
                //printf("wrong at batch %u element %u: %u %u\n", j, i, h_km[j*SABER_BYTES_CCA_DEC + i], h_k[j*SABER_KEYBYTES + i]);
                break;
            }
        }
    }


    cudaFree(d_A8);
    cudaFree(d_pk);
    cudaFree(d_c);
    cudaFree(v);
    cudaFree(pksv);
    cudaFree(sksv);
}