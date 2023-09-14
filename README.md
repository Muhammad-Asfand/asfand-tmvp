# asfand-tmvp
TMVP-based implementation of PQC schemes using GPU

# TMVP
TMVP: Acceleration of Post-quantum Cryptographic Algorithms using TMVP implementation using Tensor-cores and CUDA-cores on GPUs

#Introduction

The Tensor-cores and CUDA-cores in NVIDIA GPU are exploited to perform polynomial convolution/matrix multiplication found in several lattice-based cryptosystems. In this paper, we demonstrate two successful cases: Saber and Sable. We believe that this can benefit other similar lattice-based schemes that cannot be accelerated by NTT. This repository also contains source codes for implementing Saber and Sable parameter sets.

#How to use

There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make."

Note that you need to change the SM version in GPU to suit your device in the Makefile. The default is -arch=sm_86, which is suitable for RTX3060Ti.


