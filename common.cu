/// ****************************************************************************
/// ***************** Common utilities and  CUDA Kernels  **********************
/// ****************************************************************************

#include "common.h"
# define W_SIGN(a) ((a > 0) ? (1.0f) : (-1.0f))
#include <cublas.h>



int w_iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


int w_ipow2(int a) {
    return 1 << a;
}


int w_ilog2(int i) {
    int l = 0;
    while (i >>= 1) {
        ++l;
    }
    return l;
}


void w_swap_ptr(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}





/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_soft_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    float val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}

__global__ void w_kern_soft_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    float val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}


/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_hard_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    float val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
}

__global__ void w_kern_hard_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    float val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
}


__global__ void w_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidx < Nc && gidy < Nr) {
        int r = gidy - sr, c = gidx - sc;
        if (r < 0) r += Nr;
        if (c < 0) c += Nc;
        d_out[gidy*Nc + gidx] = d_image[r*Nc + c];
    }
}




/// ****************************************************************************
/// ******************** Common CUDA Kernels calls *****************************
/// ****************************************************************************


void w_call_soft_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs) {
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs) {
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_soft_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        w_kern_soft_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
    }
}


void w_call_hard_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs) {
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs) {
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_hard_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        w_kern_hard_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
    }
}


void w_shrink(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs) {
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs) {
        cublasSscal(Nr2*Nc2, 1.0f/(1.0f + beta), d_coeffs[0], 1);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cublasSscal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+1], 1);
        cublasSscal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+2], 1);
        cublasSscal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+3], 1);
    }
}





// if inplace = 1, the result is in "d_image" ; otherwise result is in "d_image2".
void w_call_circshift(float* d_image, float* d_image2, int Nr, int Nc, int sr, int sc, int inplace /*= 1*/) {
    // Modulus in C can be negative
    if (sr < 0) sr += Nr; // or do while loops to ensure positive numbers
    if (sc < 0) sc += Nc;
    int tpb = 16; // Threads per block
    sr = sr % Nr;
    sc = sc % Nc;
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    if (inplace) {
        cudaMemcpy(d_image2, d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image2, d_image, Nr, Nc, sr, sc);
    }
    else {
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image, d_image2, Nr, Nc, sr, sc);
    }
}




/// Creates an allocated/padded device array : [ An, H1, V1, D1, ..., Hn, Vn, Dn]
float** w_create_coeffs_buffer(int Nr, int Nc, int nlevels, int do_swt) {
    int Nr0 = Nr, Nc0 = Nc;
    if (!do_swt) { Nr0 /= 2; Nc0 /= 2; }
    float** res = (float**) calloc(3*nlevels+1, sizeof(float*));
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(float));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(float));
        cudaMalloc(&(res[i+1]), Nr*Nc*sizeof(float));
        cudaMemset(res[i+1], 0, Nr*Nc*sizeof(float));
        cudaMalloc(&(res[i+2]), Nr*Nc*sizeof(float));
        cudaMemset(res[i+2], 0, Nr*Nc*sizeof(float));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr0*Nc0*sizeof(float));
    cudaMemset(res[0], 0, Nr0*Nc0*sizeof(float));

    return res;
}

/// Deep free of wavelet coefficients
void w_free_coeffs_buffer(float** coeffs, int nlevels) {
    for (int i = 0; i < 3*nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}

/// Deep copy of wavelet coefficients. All structures must be allocated.
void w_copy_coeffs_buffer(float** dst, float** src, int Nr, int Nc, int nlevels, int do_swt) {
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+1], src[i+1], Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+2], src[i+2], Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
}



