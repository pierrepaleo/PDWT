/// ****************************************************************************
/// ***************** Common utilities and  CUDA Kernels  **********************
/// ****************************************************************************

//~ #include "utils.h"
#include "common.h"
#define W_SIGN(a) ((a > 0) ? (1.0f) : (-1.0f))
#define SQRT_2 1.4142135623730951
#include <cublas.h>


/// soft thresholding of the detail coefficients (2D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_soft_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}

/// soft thresholding of the detail coefficients (1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
// CHECKME: consider merging this kernel with the previous kernel
__global__ void w_kern_soft_thresh_1d(DTYPE* c_d, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}

/// soft thresholding of the approximation coefficients (2D and 1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_soft_thresh_appcoeffs(DTYPE* c_a, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}


/// "Cousins-threshold": set to zero the detail such as abs(detail) > abs(app)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_thresh_cousins(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = gidy*Nc + gidx;
    if (gidx < Nc && gidy < Nr) {
        DTYPE val = fabsf(c_a[tid]);
        if (val > 0) return; // CHECKME: discard non-thresholded coefficients ?
        if (fabsf(c_h[tid]) > val) c_h[tid] = 0;
        if (fabsf(c_v[tid]) > val) c_v[tid] = 0;
        if (fabsf(c_d[tid]) > val) c_d[tid] = 0;
    }
}




/// Hard thresholding of the detail coefficients (2D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void w_kern_hard_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
}


/// Hard thresholding of the detail coefficients (1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
// CHECKME: consider merging this kernel with the previous kernel
__global__ void w_kern_hard_thresh_1d(DTYPE* c_d, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
}


/// Hard thresholding of the approximation coefficients (2D and 1D)
__global__ void w_kern_hard_thresh_appcoeffs(DTYPE* c_a, DTYPE beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
}

/// Circular shift of the image (2D and 1D)
__global__ void w_kern_circshift(DTYPE* d_image, DTYPE* d_out, int Nr, int Nc, int sr, int sc) {
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

void w_call_soft_thresh(DTYPE** d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs, int normalize, int threshold_cousins) {
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (threshold_cousins) {
            puts("Warning: for now, threshold_cousins is only implemented for SWT");
            threshold_cousins = 0;
        }
        if (ndims > 1) Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs || (threshold_cousins && ndims > 1)) {
        DTYPE beta2 = beta;
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_soft_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta2, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) Nr /= 2;
            Nc /= 2;
        }
        if (normalize > 0) beta /= SQRT_2;
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        if (ndims > 1) w_kern_soft_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
        else w_kern_soft_thresh_1d<<<n_blocks, n_threads_per_block>>>(d_coeffs[i+1], beta, Nr, Nc);

        if ((threshold_cousins) && (ndims > 1)) { // no effect on 1D data
            w_kern_thresh_cousins<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc);
        }
    }
}


void w_call_hard_thresh(DTYPE** d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs, int normalize) {
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) Nr2 /= 2;
        Nc2 /= 2;
    }
    DTYPE beta2 = beta;
    if (do_thresh_appcoeffs) {
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_hard_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) Nr /= 2;
            Nc /= 2;
        }
        if (normalize > 0) beta /= SQRT_2;
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        if (ndims > 1) w_kern_hard_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
        else w_kern_hard_thresh_1d<<<n_blocks, n_threads_per_block>>>(d_coeffs[i+1], beta, Nr, Nc);
    }
}


void w_shrink(DTYPE** d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs) {
        cublas_scal(Nr2*Nc2, 1.0f/(1.0f + beta), d_coeffs[0], 1);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) Nr /= 2;
            Nc /= 2;
        }
        if (ndims == 2) {
            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+1], 1);
            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+2], 1);
            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+3], 1);
        }
        else { // 1D
            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[i+1], 1);
        }
    }
}





// if inplace = 1, the result is in "d_image" ; otherwise result is in "d_image2".
void w_call_circshift(DTYPE* d_image, DTYPE* d_image2, w_info winfos, int sr, int sc, int inplace) {
    int Nr = winfos.Nr, Nc = winfos.Nc, ndims = winfos.ndims;
    // Modulus in C can be negative
    if (sr < 0) sr += Nr; // or do while loops to ensure positive numbers
    if (sc < 0) sc += Nc;
    int tpb = 16; // Threads per block
    sr = sr % Nr;
    sc = sc % Nc;
    if (ndims == 1) sr = 0;
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    if (inplace) {
        cudaMemcpy(d_image2, d_image, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image2, d_image, Nr, Nc, sr, sc);
    }
    else {
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image, d_image2, Nr, Nc, sr, sc);
    }
}


/// Creates an allocated/padded device array : [ An, H1, V1, D1, ..., Hn, Vn, Dn]
DTYPE** w_create_coeffs_buffer(w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    int Nr0 = Nr, Nc0 = Nc;
    if (!do_swt) { Nr0 /= 2; Nc0 /= 2; }
    DTYPE** res = (DTYPE**) calloc(3*nlevels+1, sizeof(DTYPE*));
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+1]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+1], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+2]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+2], 0, Nr*Nc*sizeof(DTYPE));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr0*Nc0*sizeof(DTYPE));
    cudaMemset(res[0], 0, Nr0*Nc0*sizeof(DTYPE));

    return res;
}


/// Creates an allocated/padded device array : [ An, D1, ..., Dn]
DTYPE** w_create_coeffs_buffer_1d(w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    int Nc0 = Nc;
    if (!do_swt) Nc0 /= 2;
    DTYPE** res = (DTYPE**) calloc(nlevels+1, sizeof(DTYPE*));
    // Det coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) Nc /= 2;
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(DTYPE));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr*Nc0*sizeof(DTYPE));
    cudaMemset(res[0], 0, Nr*Nc0*sizeof(DTYPE));
    return res;
}



/// Deep free of wavelet coefficients
void w_free_coeffs_buffer(DTYPE** coeffs, int nlevels) {
    for (int i = 0; i < 3*nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}

void w_free_coeffs_buffer_1d(DTYPE** coeffs, int nlevels) {
    for (int i = 0; i < nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}


/// Deep copy of wavelet coefficients. All structures must be allocated.
void w_copy_coeffs_buffer(DTYPE** dst, DTYPE** src, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels, do_swt = winfos.do_swt;
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+1], src[i+1], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+2], src[i+2], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
}


void w_copy_coeffs_buffer_1d(DTYPE** dst, DTYPE** src, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels, do_swt = winfos.do_swt;
    // Det Coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) Nc /= 2;
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
}



///
/// ----------------------------------------------------------------------------
///




void w_add_coeffs(DTYPE** dst, DTYPE** src, w_info winfos, DTYPE alpha) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        cublas_axpy(Nr*Nc, alpha, src[i], 1, dst[i], 1);
        cublas_axpy(Nr*Nc, alpha, src[i+1], 1, dst[i+1], 1);
        cublas_axpy(Nr*Nc, alpha, src[i+2], 1, dst[i+2], 1);
    }
    // App coeff (last scale)
    cublas_axpy(Nr*Nc, alpha, src[0], 1, dst[0], 1);
}


void w_add_coeffs_1d(DTYPE** dst, DTYPE** src, w_info winfos, DTYPE alpha) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    // Det Coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) Nc /= 2;
        cublas_axpy(Nr*Nc, alpha, src[i], 1, dst[i], 1);
    }
    // App coeff (last scale)
    cublas_axpy(Nr*Nc, alpha, src[0], 1, dst[0], 1);
}



