#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "include/tv.h"
#include "include/SABLE_indcpa.cuh"
#include "include/pack.cuh"
#include "include/poly.cuh"
#include "include/SABLE_params.h"

//////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  
  uint8_t *h_m, *h_ka;
  uint8_t *h_c, *h_pk, *h_sk;
  int mode;

  mode = 0;
  cudaMallocHost((void**) &h_pk, BATCH*SABER_PUBLICKEYBYTES* sizeof(uint8_t));
  cudaMallocHost((void**) &h_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t));
  cudaMallocHost((void**) &h_sk, BATCH*SABER_SECRETKEYBYTES * sizeof(uint8_t));
  cudaMallocHost((void**) &h_m, BATCH*64 * sizeof(uint8_t));
  cudaMallocHost((void**) &h_ka, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
  //cudaMallocHost((void**) &h_kb, BATCH*SABER_KEYBYTES* sizeof(uint8_t));

    for(int i=0; i< BATCH*SABER_KEYBYTES; i++) h_ka[i] = 0;
    SABLE_enc(h_ka, h_pk, h_c, mode);
    SABLE_dec(h_c, h_pk, h_sk, h_ka, mode);

	return 0;
}


