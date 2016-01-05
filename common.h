#ifndef COMMON_H
#define COMMON_H



int w_iDivUp(int a, int b);

int w_ipow2(int a);

int w_ilog2(int i);

void w_swap_ptr(float** a, float** b);


__global__ void wavelets_kern_soft_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc);

__global__ void wavelets_kern_soft_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc);

__global__ void wavelets_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc);

#endif
