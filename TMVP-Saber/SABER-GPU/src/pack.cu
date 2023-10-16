
#include "../include/pack.cuh"

#include <stdlib.h> 


 
__device__ void BS2POLp_gpu(const uint8_t bytes[SABER_POLYCOMPRESSEDBYTES], uint16_t data[SABER_N], uint32_t stride)
{
	size_t offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*stride, bid2 = blockIdx.x*SABER_N*SABER_L;
	offset_byte = 5 * tid + bid;
	offset_data = 4 * tid + bid2;
	data[offset_data + 0] = (bytes[offset_byte + 0] & (0xff)) | ((bytes[offset_byte + 1] & 0x03) << 8);
	data[offset_data + 1] = ((bytes[offset_byte + 1] >> 2) & (0x3f)) | ((bytes[offset_byte + 2] & 0x0f) << 6);
	data[offset_data + 2] = ((bytes[offset_byte + 2] >> 4) & (0x0f)) | ((bytes[offset_byte + 3] & 0x3f) << 4);
	data[offset_data + 3] = ((bytes[offset_byte + 3] >> 6) & (0x03)) | ((bytes[offset_byte + 4] & 0xff) << 2);
}

 // void POLp2BS(uint8_t bytes[SABER_POLYCOMPRESSEDBYTES], const uint16_t data[SABER_N])
__device__ void POLp2BS_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x*SABER_N*SABER_L;

	offset_byte = 5 * tid + bid;
	offset_data = 4 * tid + bid2;
	bytes[offset_byte + 0] = (data[offset_data + 0] & (0xff));
	bytes[offset_byte + 1] = ((data[offset_data + 0] >> 8) & 0x03) | ((data[offset_data + 1] & 0x3f) << 2);
	bytes[offset_byte + 2] = ((data[offset_data + 1] >> 6) & 0x0f) | ((data[offset_data + 2] & 0x0f) << 4);
	bytes[offset_byte + 3] = ((data[offset_data + 2] >> 4) & 0x3f) | ((data[offset_data + 3] & 0x03) << 6);
	bytes[offset_byte + 4] = ((data[offset_data + 3] >> 2) & 0xff);
}

__device__ static void BS2POLq_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_SECRETKEYBYTES, bid2 =blockIdx.x*SABER_N*SABER_L ;
	offset_byte = bid + 13 * tid;
	offset_data = bid2 + 8 * tid;
	data[offset_data + 0] = (bytes[offset_byte + 0] & (0xff)) | ((bytes[offset_byte + 1] & 0x1f) << 8);
	data[offset_data + 1] = (bytes[offset_byte + 1] >> 5 & (0x07)) | ((bytes[offset_byte + 2] & 0xff) << 3) | ((bytes[offset_byte + 3] & 0x03) << 11);
	data[offset_data + 2] = (bytes[offset_byte + 3] >> 2 & (0x3f)) | ((bytes[offset_byte + 4] & 0x7f) << 6);
	data[offset_data + 3] = (bytes[offset_byte + 4] >> 7 & (0x01)) | ((bytes[offset_byte + 5] & 0xff) << 1) | ((bytes[offset_byte + 6] & 0x0f) << 9);
	data[offset_data + 4] = (bytes[offset_byte + 6] >> 4 & (0x0f)) | ((bytes[offset_byte + 7] & 0xff) << 4) | ((bytes[offset_byte + 8] & 0x01) << 12);
	data[offset_data + 5] = (bytes[offset_byte + 8] >> 1 & (0x7f)) | ((bytes[offset_byte + 9] & 0x3f) << 7);
	data[offset_data + 6] = (bytes[offset_byte + 9] >> 6 & (0x03)) | ((bytes[offset_byte + 10] & 0xff) << 2) | ((bytes[offset_byte + 11] & 0x07) << 10);
	data[offset_data + 7] = (bytes[offset_byte + 11] >> 3 & (0x1f)) | ((bytes[offset_byte + 12] & 0xff) << 5);	
}

__device__ static void BS2POLq_gpu2(uint8_t *bytes, uint16_t *data)
{
	size_t offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_L*SABER_POLYVECBYTES, bid2 =blockIdx.x*SABER_N*SABER_L*SABER_L ;
	offset_byte = bid + 13 * tid;
	offset_data = bid2 + 8 * tid;
	data[offset_data + 0] = (bytes[offset_byte + 0] & (0xff)) | ((bytes[offset_byte + 1] & 0x1f) << 8);
	data[offset_data + 1] = (bytes[offset_byte + 1] >> 5 & (0x07)) | ((bytes[offset_byte + 2] & 0xff) << 3) | ((bytes[offset_byte + 3] & 0x03) << 11);
	data[offset_data + 2] = (bytes[offset_byte + 3] >> 2 & (0x3f)) | ((bytes[offset_byte + 4] & 0x7f) << 6);
	data[offset_data + 3] = (bytes[offset_byte + 4] >> 7 & (0x01)) | ((bytes[offset_byte + 5] & 0xff) << 1) | ((bytes[offset_byte + 6] & 0x0f) << 9);
	data[offset_data + 4] = (bytes[offset_byte + 6] >> 4 & (0x0f)) | ((bytes[offset_byte + 7] & 0xff) << 4) | ((bytes[offset_byte + 8] & 0x01) << 12);
	data[offset_data + 5] = (bytes[offset_byte + 8] >> 1 & (0x7f)) | ((bytes[offset_byte + 9] & 0x3f) << 7);
	data[offset_data + 6] = (bytes[offset_byte + 9] >> 6 & (0x03)) | ((bytes[offset_byte + 10] & 0xff) << 2) | ((bytes[offset_byte + 11] & 0x07) << 10);
	data[offset_data + 7] = (bytes[offset_byte + 11] >> 3 & (0x1f)) | ((bytes[offset_byte + 12] & 0xff) << 5);	
}
// void BS2POLVECp(const uint8_t bytes[SABER_POLYVECCOMPRESSEDBYTES], uint16_t data[SABER_L][SABER_N])
__global__ void BS2POLVECp_gpu(uint8_t *bytes, uint16_t *data, uint32_t stride)
{
	size_t i;
	for (i = 0; i < SABER_L; i++)
	{
		BS2POLp_gpu(bytes + i * (SABER_EP * SABER_N / 8), data + i*SABER_N, stride);
	}
}
// void POLVECp2BS(uint8_t bytes[SABER_POLYVECCOMPRESSEDBYTES], const uint16_t data[SABER_L][SABER_N])
__global__ void POLVECp2BS_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t i;
	for (i = 0; i < SABER_L; i++)
	{
		POLp2BS_gpu(bytes + i * (SABER_EP * SABER_N / 8), data + i*SABER_N);
	}
}

// void BS2POLmsg(const uint8_t bytes[SABER_KEYBYTES], uint16_t data[SABER_N])
__global__ void BS2POLmsg_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t i;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_N, bid2 = blockIdx.x * SABER_KEYBYTES;

	// for (j = 0; j < SABER_KEYBYTES; j++)
	{
		for (i = 0; i < 8; i++)
		{
			data[bid + tid * 8 + i] = ((bytes[bid2 + tid] >> i) & 0x01);
		}
	}
}

// void POLT2BS(uint8_t bytes[SABER_SCALEBYTES_KEM], const uint16_t data[SABER_N])
__global__ void POLT2BS_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x * SABER_N;	
// #if SABER_ET == 3
// 	for (j = 0; j < SABER_N / 8; j++)
// 	{
// 		offset_byte = 3 * j;
// 		offset_data = 8 * j;
// 		bytes[offset_byte + 0] = (data[offset_data + 0] & 0x7) | ((data[offset_data + 1] & 0x7) << 3) | ((data[offset_data + 2] & 0x3) << 6);
// 		bytes[offset_byte + 1] = ((data[offset_data + 2] >> 2) & 0x01) | ((data[offset_data + 3] & 0x7) << 1) | ((data[offset_data + 4] & 0x7) << 4) | (((data[offset_data + 5]) & 0x01) << 7);
// 		bytes[offset_byte + 2] = ((data[offset_data + 5] >> 1) & 0x03) | ((data[offset_data + 6] & 0x7) << 2) | ((data[offset_data + 7] & 0x7) << 5);
// 	}
// #elif SABER_ET == 4
	// for (j = 0; j < SABER_N / 2; j++)
	{
		offset_byte = tid;
		offset_data = 2 * tid;
		bytes[bid + offset_byte] = (data[bid2 + offset_data] & 0x0f) | ((data[offset_data + 1] & 0x0f) << 4);
	}
// #elif SABER_ET == 6
// 	for (j = 0; j < SABER_N / 4; j++)
// 	{
// 		offset_byte = 3 * j;
// 		offset_data = 4 * j;
// 		bytes[offset_byte + 0] = (data[offset_data + 0] & 0x3f) | ((data[offset_data + 1] & 0x03) << 6);
// 		bytes[offset_byte + 1] = ((data[offset_data + 1] >> 2) & 0x0f) | ((data[offset_data + 2] & 0x0f) << 4);
// 		bytes[offset_byte + 2] = ((data[offset_data + 2] >> 4) & 0x03) | ((data[offset_data + 3] & 0x3f) << 2);
// 	}
// #else
// #error "Unsupported SABER parameter."
// #endif
}

// void BS2POLVECq_gpu(const uint8_t bytes[SABER_POLYVECBYTES], uint16_t data[SABER_L][SABER_N])
__global__ void BS2POLVECq_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t i;
	for (i = 0; i < SABER_L; i++)
	{
		BS2POLq_gpu(bytes + i * SABER_POLYBYTES, data+i*SABER_N);
	}
}

__global__ void BS2POLVECq_gpu2(uint8_t *bytes, uint16_t *data)
{
	size_t i, j;
	for (i = 0; i < SABER_L; i++)
	{
		for (j = 0; j < SABER_L; j++)
			BS2POLq_gpu2(bytes + i*SABER_POLYVECBYTES + j * SABER_POLYBYTES, data + i*SABER_L*SABER_N + j*SABER_N);
	}
}

	// for (i = 0; i < SABER_L; i++)
	// {
	// 	BS2POLVECq(buf + i * SABER_POLYVECBYTES, A[i]);
	// }

// void BS2POLT(const uint8_t bytes[SABER_SCALEBYTES_KEM], uint16_t data[SABER_N])
__global__ void BS2POLT_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t j, offset_byte, offset_data;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_BYTES_CCA_DEC, bid2 = blockIdx.x * SABER_N;		
// #if SABER_ET == 3
// 	for (j = 0; j < SABER_N / 8; j++)
// 	{
// 		offset_byte = 3 * j;
// 		offset_data = 8 * j;
// 		data[offset_data + 0] = (bytes[offset_byte + 0]) & 0x07;
// 		data[offset_data + 1] = ((bytes[offset_byte + 0]) >> 3) & 0x07;
// 		data[offset_data + 2] = (((bytes[offset_byte + 0]) >> 6) & 0x03) | (((bytes[offset_byte + 1]) & 0x01) << 2);
// 		data[offset_data + 3] = ((bytes[offset_byte + 1]) >> 1) & 0x07;
// 		data[offset_data + 4] = ((bytes[offset_byte + 1]) >> 4) & 0x07;
// 		data[offset_data + 5] = (((bytes[offset_byte + 1]) >> 7) & 0x01) | (((bytes[offset_byte + 2]) & 0x03) << 1);
// 		data[offset_data + 6] = ((bytes[offset_byte + 2] >> 2) & 0x07);
// 		data[offset_data + 7] = ((bytes[offset_byte + 2] >> 5) & 0x07);
// 	}
// #elif SABER_ET == 4
	offset_byte = bid + tid;
	offset_data = bid2 + 2 * tid;
	data[offset_data] = bytes[offset_byte] & 0x0f;
	data[offset_data + 1] = (bytes[offset_byte] >> 4) & 0x0f;

// #elif SABER_ET == 6
// 	for (j = 0; j < SABER_N / 4; j++)
// 	{
// 		offset_byte = 3 * j;
// 		offset_data = 4 * j;
// 		data[offset_data + 0] = bytes[offset_byte + 0] & 0x3f;
// 		data[offset_data + 1] = ((bytes[offset_byte + 0] >> 6) & 0x03) | ((bytes[offset_byte + 1] & 0x0f) << 2);
// 		data[offset_data + 2] = ((bytes[offset_byte + 1] & 0xff) >> 4) | ((bytes[offset_byte + 2] & 0x03) << 4);
// 		data[offset_data + 3] = ((bytes[offset_byte + 2] & 0xff) >> 2);
// 	}
// #else
// #error "Unsupported SABER parameter."
// #endif
}

// void POLmsg2BS(uint8_t bytes[SABER_KEYBYTES], const uint16_t data[SABER_N])
__global__ void POLmsg2BS_gpu(uint8_t *bytes, uint16_t *data)
{
	size_t i;
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_KEYBYTES, bid2 = blockIdx.x * SABER_N;		
	
	for (i = 0; i < 8; i++)
	{
		bytes[bid + tid] = bytes[bid + tid] | ((data[bid2 + tid * 8 + i] & 0x01) << i);
	}
	// bytes[bid + tid] = data[bid2 + tid];

}
    // for (i = 0; i < 32; i++) // Save hash by storing h(pk) in sk
    // h_buf[32 + i] = h_sk[SABER_SECRETKEYBYTES - 64 + i];
__global__ void copysk(uint8_t *out, uint8_t *m, uint8_t* sk)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SABER_SECRETKEYBYTES, bid2 = blockIdx.x*64, bid3 = blockIdx.x*SABER_KEYBYTES;

	// out[bid2 + tid] = tid;
	out[bid2 + tid] = m[bid3 + tid];
	out[32 + bid2 + tid] = sk[SABER_SECRETKEYBYTES - 64 + bid + tid ];
}

/* returns 0 for equal strings, 1 for non-equal strings */
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
  		// 	printf("Not same %u %u!\n", i, tid);    	
  		// }
  	}
}


/* b = 1 means mov, b = 0 means don't mov*/
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
