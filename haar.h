#ifndef HAAR_HDR
#define HAAR_HDR

#include "utils.h"

__global__ void kern_haar2d_fwd(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc);
__global__ void kern_haar2d_inv(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc);

int haar_forward2d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int haar_inverse2d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

__global__ void kern_haar1d_fwd(float* img, float* c_a, float* c_d, int Nr, int Nc);
__global__ void kern_haar1d_inv(float* img, float* c_a, float* c_d, int Nr, int Nc);

int haar_forward1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);
int haar_inverse1d(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos);

#endif



