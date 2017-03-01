#ifndef NONSEPARABLE_H
#define NONSEPARABLE_H

#include "utils.h"

DTYPE* w_outer(DTYPE* a, DTYPE* b, int len);
int w_compute_filters(const char* wname, int direction, int do_swt);
int w_set_filters_forward_nonseparable(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len);
int w_set_filters_inverse_nonseparable(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len);



__global__ void w_kern_forward(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen);
__global__ void w_kern_inverse(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2, int hlen);
int w_forward(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_inverse(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void w_kern_forward_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
__global__ void w_kern_inverse_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
int w_forward_swt(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int w_inverse_swt(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

#endif
