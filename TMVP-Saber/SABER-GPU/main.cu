#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/params.h"


// Include local CUDA header files.
#include "include/cuda_kernel.cuh"


int main(int argc, char** argv)
{   
    uint8_t mode= 1;
  uint8_t *h_m, *h_k;
  uint8_t *h_c, *h_pk, *h_sk;

  cudaMallocHost((void**) &h_pk, BATCH*SABER_PUBLICKEYBYTES* sizeof(uint8_t));
  cudaMallocHost((void**) &h_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t));
  cudaMallocHost((void**) &h_sk, BATCH*SABER_SECRETKEYBYTES * sizeof(uint8_t));
  // cudaMallocHost((void**) &h_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
  cudaMallocHost((void**) &h_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
  
    for(int i=0; i< BATCH*SABER_KEYBYTES; i++) h_k[i] = 0;
    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-m") == 0) {
            mode = atoi(argv[i + 1]);
            i += 2;
        }
        else {           
            return 0;
        }
    }
    
    saber_enc(mode, h_k, h_pk, h_c);
    saber_dec(mode, h_c, h_pk, h_sk, h_k);
    return 0;
}
