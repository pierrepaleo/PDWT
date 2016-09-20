#ifndef HAAR_HDR
#define HAAR_HDR

#include "utils.h"

__global__ void kern_haar2d_fwd(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc);
__global__ void kern_haar2d_inv(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc);

int haar_forward2d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int haar_inverse2d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

__global__ void kern_haar1d_fwd(DTYPE* img, DTYPE* c_a, DTYPE* c_d, int Nr, int Nc);
__global__ void kern_haar1d_inv(DTYPE* img, DTYPE* c_a, DTYPE* c_d, int Nr, int Nc);

int haar_forward1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);
int haar_inverse1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos);

#endif



