// The sqrt(2) factor is applied after two HAAR_*, so it becomes a 0.5 factor
#define HAAR_AVG(a, b) ((a+b))
#define HAAR_DIF(a, b) ((a-b))


// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input
__global__ void kern_haar2d_fwd(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    Nr /= 2; Nc /= 2;
    int Nc2 = Nc*2;
    if (gidy < Nr && gidx < Nc) {
        float a = img[(gidy*2)*Nc2 + (gidx*2)];
        float b = img[(gidy*2)*Nc2 + (gidx*2+1)];
        float c = img[(gidy*2+1)*Nc2 + (gidx*2)];
        float d = img[(gidy*2+1)*Nc2 + (gidx*2+1)];
        c_a[gidy* Nc + (gidx)] = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d)); // A
        c_v[gidy* Nc + (gidx)] = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d)); // V
        c_h[gidy* Nc + (gidx)] =  0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d)); // H
        c_d[gidy* Nc + (gidx)] = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d)); // D
    }
}


// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar2d_inv(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr2 = Nr*2, Nc2 = Nc*2;
    if (gidy < Nr2 && gidx < Nc2) {
        float a = c_a[(gidy/2)*Nc + (gidx/2)];
        float b = c_v[(gidy/2)*Nc + (gidx/2)];
        float c = c_h[(gidy/2)*Nc + (gidx/2)];
        float d = c_d[(gidy/2)*Nc + (gidx/2)];
        float res = 0.0f;
        int gx1 = (gidx & 1), gy1 = (gidy & 1);
        if (gx1 == 0 && gy1 == 0) res = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 1 && gy1 == 0) res = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 0 && gy1 == 1) res = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d));
        if (gx1 == 1 && gy1 == 1) res = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d));
        img[(gidy)*Nc2 + gidx] = res;
    }

}


int haar_forward2d(float* d_image, float** d_coeffs, float* d_tmp, int Nr, int Nc, int levels) {
    int Nc2 = Nc/2, Nr2 = Nr/2;
    int tpb = 16; // TODO : tune for max perfs.
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    kern_haar2d_fwd<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr2*2, Nc2*2);

    for (int i=1; i < levels; i++) {
        Nc2 /= 2;
        Nr2 /= 2;
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        kern_haar2d_fwd<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr2*2, Nc2*2);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}

int haar_inverse2d(float* d_image, float** d_coeffs, float* d_tmp, int Nr, int Nc, int levels) {
    Nr /= w_ipow2(levels);
    Nc /= w_ipow2(levels);
    int tpb = 16; // TODO : tune for max perfs.
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    for (int i = levels-1; i >= 1; i--) {
        n_blocks = dim3(w_iDivUp(Nc*2, tpb), w_iDivUp(Nr*2, tpb), 1);
        kern_haar2d_inv<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc);
        Nr *= 2;
        Nc *= 2;
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);

    // First level
    n_blocks = dim3(w_iDivUp(Nc*2, tpb), w_iDivUp(Nr*2, tpb), 1);
    kern_haar2d_inv<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc);

    return 0;
}



/// ----------------------------------------------------------------------------
/// ------------------------ 1D HAAR TRANSFORM ---------------------------------
/// ----------------------------------------------------------------------------


#define ONE_SQRT2 0.70710678118654746


// must be run with grid size = (Nc/2, Nr)  where Nr = numrows of input
__global__ void kern_haar1d_fwd(float* img, float* c_a, float* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    Nc /= 2;
    int Nc2 = Nc*2;
    if (gidy < Nr && gidx < Nc) {
        float a = img[gidy*Nc2 + (gidx*2)];
        float b = img[gidy*Nc2 + (gidx*2+1)];
        c_a[gidy* Nc + gidx] = ONE_SQRT2 * HAAR_AVG(a, b);
        c_d[gidy* Nc + gidx] = ONE_SQRT2 * HAAR_DIF(a, b);
    }
}

// must be run with grid size = (Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar1d_inv(float* img, float* c_a, float* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nc2 = Nc*2;
    if (gidy < Nr && gidx < Nc2) {
        float a = c_a[gidy*Nc + (gidx/2)];
        float b = c_d[gidy*Nc + (gidx/2)];
        float res = 0.0f;
        if ((gidx & 1) == 0) res = ONE_SQRT2 * HAAR_AVG(a, b);
        else res = ONE_SQRT2 * HAAR_DIF(a, b);
        img[gidy*Nc2 + gidx] = res;
    }
}



int haar_forward1d(float* d_image, float** d_coeffs, float* d_tmp, int Nr, int Nc, int levels) {
    int Nc2 = Nc/2;
    int tpb = 16; // TODO : tune for max perfs.
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    kern_haar1d_fwd<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], Nr, Nc2*2);

    for (int i=1; i < levels; i++) {
        Nc2 /= 2;
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr, tpb), 1);
        kern_haar1d_fwd<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[i+1], Nr, Nc2*2);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr*Nc2*sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}


int haar_inverse1d(float* d_image, float** d_coeffs, float* d_tmp, int Nr, int Nc, int levels) {
    Nc /= w_ipow2(levels);
    int tpb = 16; // TODO : tune for max perfs.
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    for (int i = levels-1; i >= 1; i--) {
        n_blocks = dim3(w_iDivUp(Nc*2, tpb), w_iDivUp(Nr, tpb), 1);
        kern_haar1d_inv<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[i+1], Nr, Nc);
        Nc *= 2;
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);

    // First level
    n_blocks = dim3(w_iDivUp(Nc*2, tpb), w_iDivUp(Nr, tpb), 1);
    kern_haar1d_inv<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], Nr, Nc);

    return 0;
}



