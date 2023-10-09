#include <stdint.h>
#ifndef INDCPA_H
#define INDCPA_H
#include "SABLE_params.h"

#define BATCH 512

void SABLE_enc(uint8_t *h_k, uint8_t *h_pk, uint8_t *h_c, int mode);

void SABLE_dec(uint8_t *h_c, uint8_t *h_pk, uint8_t *h_sk, uint8_t *h_k, int mode);


#endif

