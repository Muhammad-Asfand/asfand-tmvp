#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "SABLE_params.h"

__global__ void BS2POLVECq_gpu2(uint8_t *bytes, uint16_t *data);

__global__ void MatVecMul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_s);

__global__ void post_process(uint16_t *in);

__global__ void POLVECp2BS_gpu(uint8_t *bytes, uint16_t *data);

__global__ void BS2POLVECp_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride);

__global__ void post_process2(uint16_t *in);

__global__ void VecVecMul_Inner_gpu(uint16_t *r, uint16_t *g_a, uint16_t *g_s);

__global__ void post_process3(uint16_t *in);

__global__ void msg_unpack_encode_gpu(uint8_t *bytes, uint16_t *data);

__global__ void post_process4(uint16_t *out, uint16_t *in);

__global__ void SABER_pack_5bit(uint8_t *bytes, uint16_t *data);

__global__ void post_process5(uint8_t *out, uint8_t *in);

__global__ void BS2POLVECp_d_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride);

__global__ void post_process6(uint8_t *out, uint8_t *in);

__global__ void SABER_un_pack5bit(uint8_t *bytes, uint16_t *data);

__global__ void post_processnull(uint16_t *in);

__global__ void post_process7(uint16_t *out, uint16_t *in);

__global__ void POL2MSG(uint16_t *message_dec_unpacked, uint8_t *message_dec);

__global__ void post_process8(uint8_t *out, uint8_t *in);

__global__ void verify_gpu(uint64_t *r, uint8_t *a, uint8_t *b, size_t len);

__global__ void cmov_gpu(uint8_t *r, uint8_t *x, size_t len, uint64_t *b);

//__global__ void pol_mul(uint16_t* a, uint16_t* b, uint16_t* res, uint16_t p, uint32_t n);

__global__ void InnerProd(uint16_t* pkcl, uint16_t* skpv, uint16_t mod, uint16_t* res);

__global__ void msg_unpack_post_encode_gpu(uint16_t *data);
