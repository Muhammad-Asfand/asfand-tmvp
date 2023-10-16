#include "../include/params.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MODQ(X) ((X) & (SABER_Q-1))

__global__ void BS2POLVECp_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride);
__global__ void POLVECp2BS_gpu(uint8_t *bytes, uint16_t *data);
__global__ void BS2POLmsg_gpu(uint8_t *bytes, uint16_t *data);
__global__ void POLT2BS_gpu(uint8_t *bytes, uint16_t *data);
__global__ void BS2POLVECq_gpu(uint8_t *bytes, uint16_t *data);
__global__ void BS2POLT_gpu(uint8_t *bytes, uint16_t *data);
__global__ void POLmsg2BS_gpu(uint8_t *bytes, uint16_t *data);
__global__ void copysk(uint8_t *out, uint8_t *m, uint8_t* sk);
__global__ void verify_gpu(uint64_t *r, uint8_t *a, uint8_t *b, size_t len);
__global__ void cmov_gpu(uint8_t *r, uint8_t *x, size_t len, uint64_t *b);
__global__ void BS2POLVECq_gpu2(uint8_t *bytes, uint16_t *data);

