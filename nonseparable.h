#ifndef NONSEPARABLE_H
#define NONSEPARABLE_H

#include "utils.h"

float* w_outer(float* a, float* b, int len);
int w_compute_filters(const char* wname, int direction, int do_swt);

__global__ void w_kern_forward(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen);
__global__ void w_kern_inverse(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen);
int w_forward(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_inverse(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

__global__ void w_kern_forward_swt(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_inverse_swt(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen, int level);
int w_forward_swt(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int w_inverse_swt(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

#endif
