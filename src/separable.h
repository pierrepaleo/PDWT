#ifndef SEPARABLE_H
#define SEPARABLE_H
#include "utils.h"


int w_compute_filters_separable(const char* wname, int do_swt);
__global__ void w_kern_forward_pass1(float* img, float* tmp_a1, float* tmp_a2, int Nr, int Nc, int hlen);
__global__ void w_kern_forward_pass2(float* tmp_a1, float* tmp_a2, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen);
int w_forward_separable(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_forward_separable_1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

__global__ void w_kern_inverse_pass1(float* c_a, float* c_h, float* c_v, float* c_d, float* tmp1, float* tmp2, int Nr, int Nc, int hlen);
__global__ void w_kern_inverse_pass2(float* tmp1, float* tmp2, float* img, int Nr, int Nc, int hlen);
int w_inverse_separable(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_inverse_separable_1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

__global__ void w_kern_forward_swt_pass1(float* img, float* tmp_a1, float* tmp_a2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_forward_swt_pass2(float* tmp_a1, float* tmp_a2, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen, int level);
int w_forward_swt_separable(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_forward_swt_separable_1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

__global__ void w_kern_inverse_swt_pass1(float* c_a, float* c_h, float* c_v, float* c_d, float* tmp1, float* tmp2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_inverse_swt_pass2(float* tmp1, float* tmp2, float* img, int Nr, int Nc, int hlen, int level);
int w_inverse_swt_separable(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_inverse_swt_separable_1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);



#endif

