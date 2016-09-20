#ifndef COMMON_H
#define COMMON_H

#include "utils.h"

// For all architectures, constant mem is limited to 64 KB.
// Here we limit the filter size to 40x40 coefficients => 25.6KB
// If you know the max width of filters used in practice, it might be interesting to define it here
// since MAX_FILTER_WIDTH * MAX_FILTER_WIDTH * 4   elements are transfered at each transform scale
//
// There are two approaches for inversion :
//  - compute the inverse filters into the previous constant arrays, before W.inverse(). It is a little slower.
//  - pre-compute c_kern_inv_XX once for all... faster, but twice more memory is used
#define MAX_FILTER_WIDTH 40

__constant__ float c_kern_L[MAX_FILTER_WIDTH];
__constant__ float c_kern_H[MAX_FILTER_WIDTH];
__constant__ float c_kern_IL[MAX_FILTER_WIDTH];
__constant__ float c_kern_IH[MAX_FILTER_WIDTH];

__constant__ float c_kern_LL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ float c_kern_LH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ float c_kern_HL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
__constant__ float c_kern_HH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];



__global__ void w_kern_soft_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc);

__global__ void w_kern_soft_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc);

__global__ void w_kern_hard_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc);

__global__ void w_kern_hard_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc);

__global__ void w_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc);


void w_call_soft_thresh(float** d_coeffs, float beta, w_info winfos, int do_thresh_appcoeffs, int normalize, int threshold_cousins);

void w_call_hard_thresh(float** d_coeffs, float beta, w_info winfos, int do_thresh_appcoeffs, int normalize);

void w_shrink(float** d_coeffs, float beta, w_info winfos, int do_thresh_appcoeffs);

void w_call_circshift(float* d_image, float* d_image2, w_info winfos, int sr, int sc, int inplace = 1);

float** w_create_coeffs_buffer(w_info winfos);
void w_free_coeffs_buffer(float** coeffs, int nlevels);
void w_copy_coeffs_buffer(float** dst, float** src, w_info winfos);

float** w_create_coeffs_buffer_1d(w_info winfos);
void w_free_coeffs_buffer_1d(float** coeffs, int nlevels);
void w_copy_coeffs_buffer_1d(float** dst, float** src, w_info winfos);

__global__ void w_kern_hard_thresh_1d(float* c_d, float beta, int Nr, int Nc);
__global__ void w_kern_soft_thresh_1d(float* c_d, float beta, int Nr, int Nc);

void w_add_coeffs(float** dst, float** src, w_info winfos, float alpha=1.0f);
void w_add_coeffs_1d(float** dst, float** src, w_info winfos, float alpha=1.0f);










#endif
