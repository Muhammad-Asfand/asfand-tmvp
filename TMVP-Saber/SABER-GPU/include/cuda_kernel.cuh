#define BATCH 1
// #define DEBUG 		// To check the encryption/decryption results
// #define MEAS_MV		// Measure matrix-vector multiplication
// #define MEAS_IP		// Measure inner product

void saber_enc(uint8_t mode, uint8_t *h_k, uint8_t *h_pk, uint8_t *h_c) ;
void saber_dec(uint8_t mode, uint8_t *h_c, uint8_t *h_pk, uint8_t *h_sk, uint8_t *h_k) ;



