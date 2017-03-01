#ifndef SEPARABLE_H
#define SEPARABLE_H
#include "utils.h"

int w_compute_filters_separable(const char* wname, int do_swt);
int w_set_filters_forward(DTYPE* filter1, DTYPE* filter2, uint len);
int w_set_filters_inverse(DTYPE* filter1, DTYPE* filter2, uint len);


__global__ void w_kern_forward_pass1(DTYPE* img, DTYPE* tmp_a1, DTYPE* tmp_a2, int Nr, int Nc, int hlen);
__global__ void w_kern_forward_pass2(DTYPE* tmp_a1, DTYPE* tmp_a2, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen);
int w_forward_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_forward_separable_1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void w_kern_inverse_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int Nr2, int hlen);
__global__ void w_kern_inverse_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int Nc2, int hlen);
int w_inverse_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_inverse_separable_1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void w_kern_forward_swt_pass1(DTYPE* img, DTYPE* tmp_a1, DTYPE* tmp_a2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_forward_swt_pass2(DTYPE* tmp_a1, DTYPE* tmp_a2, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
int w_forward_swt_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_forward_swt_separable_1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void w_kern_inverse_swt_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_inverse_swt_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int hlen, int level);
int w_inverse_swt_separable(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_inverse_swt_separable_1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);



#endif

