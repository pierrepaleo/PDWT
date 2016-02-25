#ifndef COMMON_H
#define COMMON_H



int w_iDivUp(int a, int b);

int w_ipow2(int a);

int w_ilog2(int i);

void w_swap_ptr(float** a, float** b);


__global__ void w_kern_soft_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc);

__global__ void w_kern_soft_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc);

__global__ void w_kern_hard_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc);

__global__ void w_kern_hard_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc);

__global__ void w_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc);


void w_call_soft_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs, int ndim);

void w_call_hard_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs, int ndim);

void w_shrink(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs, int ndim);

void w_call_circshift(float* d_image, float* d_image2, int Nr, int Nc, int sr, int sc, int inplace = 1, int ndim = 2);

float** w_create_coeffs_buffer(int Nr, int Nc, int nlevels, int do_swt);

void w_free_coeffs_buffer(float** coeffs, int nlevels);

void w_copy_coeffs_buffer(float** dst, float** src, int Nr, int Nc, int nlevels, int do_swt);

// ---

float** w_create_coeffs_buffer_1d(int Nr, int Nc, int nlevels, int do_swt);
void w_free_coeffs_buffer_1d(float** coeffs, int nlevels);
void w_copy_coeffs_buffer_1d(float** dst, float** src, int Nr, int Nc, int nlevels, int do_swt);

__global__ void w_kern_hard_thresh_1d(float* c_d, float beta, int Nr, int Nc);
__global__ void w_kern_soft_thresh_1d(float* c_d, float beta, int Nr, int Nc);




#endif
