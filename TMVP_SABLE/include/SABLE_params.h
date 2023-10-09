//#include "api.h"

#ifndef PARAMS_H
#define PARAMS_H

#define Saber_type 2

#if Saber_type == 1
	#define SABER_K 2
	#define SABER_EP 9
	#define SABER_P 512 
	#define SABER_ET 3

#elif Saber_type == 2
	#define SABER_K 3
	#define SABER_EP 9
	#define SABER_P 512 
	#define SABER_ET 5

#elif Saber_type == 3
	#define SABER_K 4
	#define SABER_EP 10
	#define SABER_P 1024 
	#define SABER_ET 3
#endif
#define SABER_MU 2
#define SABER_EQ 11

#define SABER_N 256
#define SABER_Q 2048 


#define SABER_SEEDBYTES       32
#define SABER_NOISESEEDBYTES  32
#define SABER_COINBYTES       32
#define SABER_KEYBYTES        32

#define SABER_HASHBYTES       32

#define SABER_POLYBYTES       352 //11*256/8 

#define SABER_POLYVECBYTES    (SABER_K * SABER_POLYBYTES) 

#if Saber_type == 1
	#define SABER_POLYVECCOMPRESSEDBYTES (SABER_K * 288) //9*256/8 NOTE : changed till here due to parameter adaptation
#elif Saber_type == 2
	#define SABER_POLYVECCOMPRESSEDBYTES (SABER_K * 288) //9*256/8 NOTE : changed till here due to parameter adaptation
#elif Saber_type == 3
	#define SABER_POLYVECCOMPRESSEDBYTES (SABER_K * 320) //10*256/8 NOTE : changed till here due to parameter adaptation
#endif  

#define SABER_CIPHERTEXTBYTES (SABER_POLYVECCOMPRESSEDBYTES)

#define SABER_SCALEBYTES (SABER_DELTA*SABER_N/8)

#define SABER_SCALEBYTES_KEM ((SABER_ET)*SABER_N/8)

#define SABER_INDCPA_PUBLICKEYBYTES (SABER_POLYVECCOMPRESSEDBYTES + SABER_SEEDBYTES)
#define SABER_INDCPA_SECRETKEYBYTES (SABER_K * 64)//2*256/8

#define SABER_PUBLICKEYBYTES (SABER_INDCPA_PUBLICKEYBYTES)

#define SABER_SECRETKEYBYTES (SABER_INDCPA_SECRETKEYBYTES +  SABER_INDCPA_PUBLICKEYBYTES + SABER_HASHBYTES + SABER_KEYBYTES)

#define SABER_BYTES_CCA_DEC   (SABER_POLYVECCOMPRESSEDBYTES + SABER_SCALEBYTES_KEM) /* Second part is for Targhi-Unruh */

#define buf_size (SABER_MU*SABER_N*SABER_K/8)

#define SABER_A8 (SABER_K*SABER_EQ*SABER_N/8)

#define SABER_POLYCOINBYTES (SABER_MU * SABER_N / 8)

#endif
