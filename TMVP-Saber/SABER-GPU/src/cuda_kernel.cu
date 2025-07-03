// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/params.h"
#include "../include/poly.cuh"
#include "../include/pack.cuh"
#include "../include/tv.h"

#include <stdio.h>
#include <stdint.h>
#define MOD256(X) ((X) & (256-1))
#define MOD65536(X) ((X) & (65536-1))


void saber_enc(uint8_t mode, uint8_t *h_k, uint8_t *h_pk, uint8_t *h_c)  {
    uint8_t *d_pk, *d_c, *d_m, *d_kr, *d_buf, *d_k, *d_A8;
    uint8_t *h_buf, *h_m, *h_A8;
    int i, j, k;
    uint16_t *h_sp, *h_bp, *h_b, *h_vp, *h_A; 
    uint16_t *d_A, *d_sp, *d_bp, *d_b, *d_vp; 
    uint16_t *d_mp, *arr; 
    half *h_da, *h_skpv1;
    half *a1, *b1, *a2, *b2, *a3, *b3;
    float *c_wmma2, *c_wmma1, *c_wmma3, *c_wmma4;
    uint16_t *ac1, *bc1, *ac2, *bc2, *ac3, *bc3;
    uint16_t *p1, *p2, *p3;

    short2 *d_packed_A1, *d_packed_A2;
    char4 *d_packed_b, *h_packed_b;
    cudaEvent_t start, stop, startIP, startMV, stopIP, stopMV;
    float elapsed, elapsedIP, elapsedMV;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventCreate(&startIP);    cudaEventCreate(&stopIP);
    cudaEventCreate(&startMV);    cudaEventCreate(&stopMV);
    cudaMallocHost((void**) &h_buf, BATCH*64* sizeof(uint8_t));    
    cudaMallocHost((void**) &h_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));    
    cudaMallocHost((void**) &h_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));   
    cudaMallocHost((void**) &h_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));        
    cudaMallocHost((void**) &h_vp, BATCH*SABER_N* sizeof(uint16_t));  
    cudaMallocHost((void**) &h_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));

    cudaMalloc((void**) &ac1, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc1, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));
    cudaMalloc((void**) &ac2, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc2, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));
    cudaMalloc((void**) &ac3, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));   
    cudaMalloc((void**) &bc3, (SABER_N/2)*(SABER_N/2) * sizeof(uint16_t));

    cudaMalloc((void**) &p1, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p2, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p3, BATCH*(SABER_N)* sizeof(uint16_t));


    cudaMalloc((void**) &a1, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b1, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a2, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b2, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &a3, SABER_N/2 * SABER_N/2 * sizeof(half));   
    cudaMalloc((void**) &b3, SABER_N/2 * SABER_N/2 * sizeof(half));
    cudaMalloc((void**) &h_skpv1, SABER_N * SABER_N * sizeof(half));
    cudaMalloc((void**) &h_da, SABER_N * SABER_N * sizeof(half));
    cudaMalloc((void**) &c_wmma1, SABER_N/2*SABER_N/2 * sizeof(float));   
    cudaMalloc((void**) &c_wmma2, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma3, SABER_N/2*SABER_N/2 * sizeof(float));
    cudaMalloc((void**) &c_wmma4, SABER_N*SABER_N * sizeof(float));

    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));    
    cudaMalloc((void**) &d_A, BATCH*SABER_L*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));    
    cudaMalloc((void**) &d_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));       
    cudaMalloc((void**) &d_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &arr, SABER_N*SABER_N* sizeof(uint16_t));       
    cudaMalloc((void**) &d_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES* sizeof(uint8_t));   
    cudaMalloc((void**) &d_packed_A1, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));   
    cudaMalloc((void**) &d_packed_A2, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));    
    cudaMalloc((void**) &d_packed_b, BATCH*SABER_L*SABER_N* sizeof(char4));    
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));     
    cudaMalloc((void**) &d_vp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_mp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_A8, BATCH*SABER_L * SABER_POLYVECBYTES* sizeof(uint8_t));       

    // Public and private key are from test vectors
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk[i];
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_KEYBYTES; i++) h_m[j*SABER_KEYBYTES + i] = m_tv[i];      
    for(j=0; j<BATCH; j++) for(i=0; i<64; i++) h_buf[j*64 + i] = buf_tv[i];  // from randombytes()

    uint32_t threads = 32 * ((SABER_N/2)/WMMA_M)*((SABER_N/2)/WMMA_M);// each warp computes 16x16 matrix
    uint32_t blocks = 1;
    if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }

             
    cudaMemcpy(d_buf, h_buf, BATCH*64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
 
    cudaEventRecord(start);

    sha3_256_gpu<<<1,BATCH>>>(d_buf, d_buf, 32, 64, 64); 
    sha3_256_gpu<<<1,BATCH>>>(d_buf + 32, d_pk, SABER_INDCPA_PUBLICKEYBYTES, SABER_INDCPA_PUBLICKEYBYTES, 64); 
    sha3_512_gpu<<<1,BATCH>>>(d_kr, d_buf, 64);

    // start of indcpa_kem_enc

        cudaEventRecord(startMV);  

    GenSecret_gpu<<<1,BATCH>>>(d_sp, d_kr + 32);    
    // GenMatrix_gpu2<<<1,BATCH>>>(d_A, d_pk + SABER_POLYVECCOMPRESSEDBYTES);   
    shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_L * SABER_POLYVECBYTES, SABER_L * SABER_POLYVECBYTES);
    BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_A);    
      
    MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_bp, d_A, d_sp);     
  
    post_process<<<BATCH, SABER_N>>>(d_bp);
    // // POLVECp2BS(ciphertext, bp);
    POLVECp2BS_gpu<<<BATCH, SABER_N / 4>>>(d_c, d_bp);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_pk, d_b, SABER_INDCPA_PUBLICKEYBYTES);


#ifdef MEAS_IP    
    cudaEventRecord(startIP);         
#endif 
if(mode==0)
{
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, d_sp + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(a1, b1, a2, b2, a3, b3, arr, d_b + i*SABER_N); 
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(d_vp, c_wmma1, c_wmma2, c_wmma3); 
    }    
}
else if(mode==1)
{
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, d_sp + i*SABER_N);
        submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(ac1, bc1, ac2, bc2, ac3, bc3, arr, d_b + i*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>>(ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, d_vp);
    }
}
else
{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_vp, d_b, d_sp);
}
#ifdef MEAS_IP    
    cudaEventRecord(stopIP);      
    cudaEventSynchronize(stopIP);    
    cudaEventElapsedTime(&elapsedIP, startIP, stopIP);    
    if(mode==0)
        printf("Inner Product: DPSaber %.4f ms TP %.0f /s\n", elapsedIP, BATCH*1000/elapsedIP);
    elseif(mode==0)
        printf("Inner Product: DPSaber %.4f ms TP %.0f /s\n", elapsedIP, BATCH*1000/elapsedIP);
    else
        printf("Inner Product: INT32 %.4f ms TP %.0f /s\n", elapsedIP, BATCH*1000/elapsedIP);       
#endif   

    BS2POLmsg_gpu<<<BATCH, SABER_KEYBYTES>>>(d_m, d_mp);
    post_process2<<<BATCH, SABER_N>>>(d_vp, d_mp);
    POLT2BS_gpu<<<BATCH, SABER_N / 2>>>(d_c+SABER_POLYVECCOMPRESSEDBYTES, d_vp);
    // end of indcpa_kem_enc

    cudaEventRecord(stopMV);      
    cudaEventSynchronize(stopMV);    
    cudaEventElapsedTime(&elapsedMV, startMV, stopMV); 

    sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64); 
    sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, 32); 
    
    cudaEventRecord(stop);      
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaMemcpy(h_c, d_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyDeviceToHost);           
    cudaMemcpy(h_k, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
        
    if(mode==0)
        printf("Encap: TMVP-Tensor %.4f ms TP %.0f Encap/s TP %.0f Encryption/s\n", elapsed, BATCH*1000/elapsed , BATCH*1000/elapsedMV);
    else if(mode==1)
        printf("Encap: TMVP-CUDA %.4f ms TP %.0f Encap/s TP %.0f Encryption/s\n", elapsed, BATCH*1000/elapsed , BATCH*1000/elapsedMV);
    else
        printf("Encap: TMVP-Schoolbook %.4f ms TP %.0f Encap/s TP %.0f Encryption/s\n", elapsed, BATCH*1000/elapsed , BATCH*1000/elapsedMV);
#ifdef DEBUG
    printf("\n h_k:\n"); for(j=0; j<2; j++) {printf("\nbatch: %u\n", j); for(i=0; i<SABER_KEYBYTES; i++) printf("%u ", h_k[j*SABER_KEYBYTES + i]);}
    printf("\n h_c:\n"); for(k=0; k<2; k++) {printf("\nbatch %u\n", k); for(i=0; i<SABER_BYTES_CCA_DEC; i++) printf("%u ", h_c[k*SABER_BYTES_CCA_DEC + i]);}        
#endif
    
    cudaDeviceSynchronize();
    cudaFree(d_pk); cudaFree(d_c);  cudaFree(d_m);
    cudaFreeHost(h_sp);  cudaFreeHost(h_bp); 
    cudaFreeHost(h_b); cudaFreeHost(h_vp); 
}

void saber_dec(uint8_t mode, uint8_t *h_c, uint8_t *h_pk, uint8_t *h_sk, uint8_t *h_k) 
{
    uint8_t *h_m, *h_buf, *h_cCompare;
    uint8_t *d_sk, *d_c, *d_m, *d_kr, *d_buf, *d_pk, *d_cCompare, *d_k, *d_A8;
    uint16_t *h_s, *h_b, *h_v, *d_A;
    uint16_t *d_s, *d_b, *d_v, *d_cm, *d_sp, *d_bp, *d_vp, *d_mp, *arr, *d_arr;
    uint64_t *d_r;
    int i, j, k;
    half *h_fp16, *h_skpv1;
    float *c_wmma4;

    half *a1, *b1, *a2, *b2, *a3, *b3;
    float *c_wmma2, *c_wmma1, *c_wmma3;

    half *d_a1, *d_b1, *d_a2, *d_b2, *d_a3, *d_b3;
    float *d_wmma2, *d_wmma1, *d_wmma3;
    uint16_t *p1, *p2, *p3, *d_p1, *d_p2, *d_p3;
    uint16_t *ac1, *bc1, *ac2, *bc2, *ac3, *bc3;
    uint16_t *d_ac1, *d_bc1, *d_ac2, *d_bc2, *d_ac3, *d_bc3;

    short2 *d_packed_A1, *d_packed_A2;
    char4 *d_packed_b, *h_packed_b;
    cudaEvent_t start, stop, startIP, stopIP;
    float elapsed, elapsedIP;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventCreate(&startIP);    cudaEventCreate(&stopIP);
    cudaMallocHost((void**) &h_s, BATCH*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMallocHost((void**) &h_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));       
    cudaMallocHost((void**) &h_v, BATCH*SABER_N* sizeof(uint16_t));
    cudaMallocHost((void**) &h_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMallocHost((void**) &h_buf, BATCH*64* sizeof(uint8_t));  
    cudaMallocHost((void**) &h_cCompare, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));

    cudaMalloc((void**) &h_fp16, SABER_N * SABER_N * sizeof(half));   
    cudaMalloc((void**) &h_skpv1, SABER_N * SABER_N * sizeof(half));

    cudaMalloc((void**) &p1, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p2, BATCH*(SABER_N)* sizeof(uint16_t));
    cudaMalloc((void**) &p3, BATCH*(SABER_N)* sizeof(uint16_t));

    cudaMalloc((void**) &d_p1, BATCH*(SABER_N/2)* sizeof(uint16_t));
    cudaMalloc((void**) &d_p2, BATCH*(SABER_N/2)* sizeof(uint16_t));
    cudaMalloc((void**) &d_p3, BATCH*(SABER_N/2)* sizeof(uint16_t));


    cudaMalloc((void**) &ac1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &ac2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &ac3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &bc3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));

    cudaMalloc((void**) &d_ac1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc1, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &d_ac2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc2, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));
    cudaMalloc((void**) &d_ac3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));   
    cudaMalloc((void**) &d_bc3, SABER_N/2 * SABER_N/2 * sizeof(uint16_t));

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

    cudaMalloc((void**) &d_A8, BATCH*SABER_L * SABER_POLYVECBYTES* sizeof(uint8_t));       
    cudaMalloc((void**) &d_r, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint64_t));  
    cudaMalloc((void**) &d_vp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_mp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &arr, SABER_N*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_arr, SABER_N*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES* sizeof(uint8_t));   
    cudaMalloc((void**) &d_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));  
    cudaMalloc((void**) &d_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));     
    cudaMalloc((void**) &d_A, BATCH*SABER_L*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_sk, BATCH*SABER_SECRETKEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));
    cudaMalloc((void**) &d_cCompare, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));    
    cudaMalloc((void**) &d_s, BATCH*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t)); 
    cudaMalloc((void**) &d_v, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_cm, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_packed_A1, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));   
    cudaMalloc((void**) &d_packed_A2, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));    
    cudaMalloc((void**) &d_packed_b, BATCH*SABER_L*SABER_N/2* sizeof(char4));     
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));     
    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));        
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMemset(d_m, 0, BATCH*SABER_KEYBYTES* sizeof(uint8_t)); 

        // Public and private key are from test vectors
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_SECRETKEYBYTES; i++) h_sk[j*SABER_SECRETKEYBYTES + i] = sk_tv[i];
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk[i];

    uint32_t threads = 32 * ((SABER_N/2)/WMMA_M)*((SABER_N/2)/WMMA_M);// each warp computes 16x16 matrix
    uint32_t blocks = 1;
    if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }
    
         
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_s, h_s, BATCH*SABER_L*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, BATCH*SABER_L*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, h_sk, BATCH*SABER_SECRETKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    cudaEventRecord(startIP);
    // start of indcpa_kem_dec
    BS2POLVECq_gpu<<<BATCH, SABER_N/8>>>(d_sk, d_s);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_c, d_b, SABER_BYTES_CCA_DEC);

if(mode==0)
{
    
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(d_arr, d_s + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(d_a1, d_b1, d_a2, d_b2, d_a3, d_b3, d_arr, d_b + i*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (d_a1, d_b1, d_a2, d_b2, d_a3, d_b3, d_wmma1, d_wmma2, d_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(d_v, d_wmma1, d_wmma2, d_wmma3); 
    }
}
else if(mode==1)
{
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(d_arr, d_s + i*SABER_N);
        submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(ac1, bc1, ac2, bc2, ac3, bc3, d_arr, d_b + i*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>>(ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, d_v);
    }
}
else{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_v, d_b, d_s);
}
    BS2POLT_gpu<<<BATCH, SABER_N / 2>>>(d_c + SABER_POLYVECCOMPRESSEDBYTES, d_cm);
    post_process3<<<BATCH, SABER_N >>>(d_v, d_cm);
    POLmsg2BS_gpu<<<BATCH, SABER_KEYBYTES >>>(d_m, d_v);

    cudaEventRecord(stopIP);      
    cudaEventSynchronize(stopIP);    
    cudaEventElapsedTime(&elapsedIP, startIP, stopIP);

    // end of indcpa_kem_dec
  // Multitarget countermeasure for coins + contributory KEM
    copysk<<<BATCH, 32>>>(d_buf, d_m, d_sk);
    sha3_512_gpu<<<1,BATCH>>>(d_kr, d_buf, 64);

    // ************* start of indcpa_kem_enc *************
    // GenMatrix_gpu2<<<1,BATCH>>>(d_A, d_pk + SABER_POLYVECCOMPRESSEDBYTES);
    shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_L * SABER_POLYVECBYTES, SABER_L * SABER_POLYVECBYTES);
    BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_A);    
    GenSecret_gpu<<<1,BATCH>>>(d_sp, d_kr + 32);    
    
if(mode==0)    
{
    for (int j = 0; j < SABER_L; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, d_sp + j*SABER_N);
        submatrix_m1_tensor <<< SABER_N/2, SABER_N/2 >>>(a1, a2, a3, arr);
    for (int i = 0; i < SABER_L; i++){
        submatrix_m2_tensor <<< SABER_N/2, SABER_N/2 >>>(b1, b2, b3, d_A + i*SABER_N*SABER_L + j*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP_m<<< BATCH, SABER_N/2 >>>(d_bp + i*SABER_N, c_wmma1, c_wmma2, c_wmma3);
    }}
    //MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_bp, d_A, d_sp);  
}
else if(mode==1)
{
    for (int j = 0; j < SABER_L; j++){
        convertnegacyclictest2<<< SABER_N, SABER_N >>>(arr, d_sp + j*SABER_N);
        submatrix_m2 <<< SABER_N/2, SABER_N/2 >>>(ac1, ac2, ac3, arr);
    for (int i = 0; i < SABER_L; i++){
        submatrix_m1 <<< BATCH, SABER_N/2 >>>(bc1, bc2, bc3, d_A + i*SABER_N*SABER_L + j*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>> (ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cudaq <<< BATCH, SABER_N/2 >>>(p1, p2, p3, d_bp + i*SABER_N);
    }} 
}
else{
    MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_bp, d_A, d_sp);     
}
    post_process<<<BATCH, SABER_N>>>(d_bp);    
    POLVECp2BS_gpu<<<BATCH, SABER_N / 4>>>(d_cCompare, d_bp);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_pk, d_b, SABER_INDCPA_PUBLICKEYBYTES);

if(mode==0)
{
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, d_sp + i*SABER_N);
        submatrix <<< SABER_N/2, SABER_N/2 >>>(a1, b1, a2, b2, a3, b3, arr, d_b + i*SABER_N);
        wmma_ker_padding2<<< blocks, threads >>> (a1, b1, a2, b2, a3, b3, c_wmma1, c_wmma2, c_wmma3);
        convertFp32ToU16modP<<<BATCH, SABER_N/2 >>>(d_vp, c_wmma1, c_wmma2, c_wmma3); 
    }
}
else if(mode==1)
{
    for (int i = 0; i < SABER_L; i++)
    {
        convertnegacyclictest<<< SABER_N, SABER_N >>>(arr, d_sp + i*SABER_N);
        submatrix_cuda <<< SABER_N/2, SABER_N/2 >>>(ac1, bc1, ac2, bc2, ac3, bc3, arr, d_b + i*SABER_N);
        matvecp_cuda <<< BATCH, SABER_N >>>(ac1, bc1, ac2, bc2, ac3, bc3, p1, p2, p3);
        matvecout_cuda <<< BATCH, SABER_N/2 >>>(p1, p2, p3, d_vp);
    }
}
else{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_vp, d_b, d_sp);
}
    BS2POLmsg_gpu<<<BATCH, SABER_KEYBYTES>>>(d_m, d_mp);
    post_process2<<<BATCH, SABER_N>>>(d_vp, d_mp);
    POLT2BS_gpu<<<BATCH, SABER_N / 2>>>(d_cCompare+SABER_POLYVECCOMPRESSEDBYTES, d_vp);
    // ************* end of indcpa_kem_enc *************
    // printf("SABER_BYTES_CCA_DEC: %u\n", SABER_BYTES_CCA_DEC);
    verify_gpu<<<BATCH, SABER_N>>>(d_r, d_c, d_cCompare, SABER_BYTES_CCA_DEC);
    // overwrite coins in kr with h(c)
    sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64); 
    // hash concatenation of pre-k and h(c) to k
    sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, SABER_KEYBYTES); 
    cmov_gpu<<<BATCH,SABER_N>>>(d_kr, d_sk+ SABER_SECRETKEYBYTES - SABER_KEYBYTES, SABER_KEYBYTES, d_r);

    cudaEventRecord(stop);      
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);    

    cudaMemcpy(h_m, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);       


    if(mode==0)
        printf("Decap: TMVP-Tensor %.4f ms TP %.0f Decap/s TP %.0f Decryption/s\n", elapsed, BATCH*1000/elapsed, BATCH*1000/elapsedIP);
    else if(mode==1)
        printf("Decap: TMVP-CUDA %.4f ms TP %.0f Decap/s TP %.0f Decryption/s\n", elapsed, BATCH*1000/elapsed, BATCH*1000/elapsedIP);
    else
        printf("Decap: TMVP-Schoolbook %.4f ms TP %.0f Decap/s TP %.0f Decryption/s\n", elapsed, BATCH*1000/elapsed, BATCH*1000/elapsedIP);    
#ifdef DEBUG
    //printf("\n h_m:\n"); for(k=0; k<2; k++) {printf("\nbatch %u\n", k); for(i=0; i<SABER_KEYBYTES; i++) printf("%u ", h_m[k*SABER_KEYBYTES + i]);}    
    for(j=0; j<BATCH; j++)
    {
        for(i=0; i<SABER_KEYBYTES; i++)
        {
            if(h_m[j*SABER_KEYBYTES + i]!=h_k[j*SABER_KEYBYTES + i]){
                //printf("wrong at batch %u element %u: %u %u\n", j, i, h_m[j*SABER_BYTES_CCA_DEC + i], h_k[j*SABER_KEYBYTES + i]);
                break;
            }
        }
    }
#endif
    cudaFreeHost(h_s); cudaFreeHost(h_b);  cudaFreeHost(h_v);     
}
