#include "haar.h"
#include "utils.h"

// The sqrt(2) factor is applied after two HAAR_*, so it becomes a 0.5 factor
#define HAAR_AVG(a, b) ((a+b))
#define HAAR_DIF(a, b) ((a-b))


// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input
__global__ void kern_haar2d_fwd(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr_is_odd = (Nr & 1);
    int Nc_is_odd = (Nc & 1);
    int Nr2 = (Nr + Nr_is_odd)/2;
    int Nc2 = (Nc + Nc_is_odd)/2;
    if (gidy < Nr2 && gidx < Nc2) {

        // for odd N, image is virtually extended by repeating the last element
        int posx0 = 2*gidx;
        int posx1 = 2*gidx+1;
        if ((Nc_is_odd) && (posx1 == Nc)) posx1--;
        int posy0 = 2*gidy;
        int posy1 = 2*gidy+1;
        if ((Nr_is_odd) && (posy1 == Nr)) posy1--;

        DTYPE a = img[posy0*Nc + posx0];
        DTYPE b = img[posy0*Nc + posx1];
        DTYPE c = img[posy1*Nc + posx0];
        DTYPE d = img[posy1*Nc + posx1];

        c_a[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d)); // A
        c_v[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d)); // V
        c_h[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d)); // H
        c_d[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d)); // D
    }
}


// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar2d_inv(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr2 && gidx < Nc2) {
        DTYPE a = c_a[(gidy/2)*Nc + (gidx/2)];
        DTYPE b = c_v[(gidy/2)*Nc + (gidx/2)];
        DTYPE c = c_h[(gidy/2)*Nc + (gidx/2)];
        DTYPE d = c_d[(gidy/2)*Nc + (gidx/2)];
        DTYPE res = 0.0f;
        int gx1 = (gidx & 1), gy1 = (gidy & 1);
        if (gx1 == 0 && gy1 == 0) res = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 1 && gy1 == 0) res = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 0 && gy1 == 1) res = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d));
        if (gx1 == 1 && gy1 == 1) res = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d));
        img[gidy*Nc2 + gidx] = res;
    }

}


int haar_forward2d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr2;
    w_div2(&Nc2); w_div2(&Nr2);
    int tpb = 16; // TODO : tune for max perfs.
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    kern_haar2d_fwd<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc);

    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2); w_div2(&Nr2);
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        kern_haar2d_fwd<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr2_old, Nc2_old);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    return 0;
}

int haar_inverse2d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels;

    // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNr[levels+1]; tNr[0] = Nr;
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNr[i] = tNr[i-1];
        tNc[i] = tNc[i-1];
        w_div2(tNr + i);
        w_div2(tNc + i);
    }

    int tpb = 16; // TODO : tune for max perfs.
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    for (int i = levels-1; i >= 1; i--) {
        n_blocks = dim3(w_iDivUp(tNc[i], tpb), w_iDivUp(tNr[i], tpb), 1);
        kern_haar2d_inv<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], tNr[i+1], tNc[i+1], tNr[i], tNc[i]);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, tNr[1]*tNc[1]*sizeof(DTYPE), cudaMemcpyDeviceToDevice);

    // First level
    n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    kern_haar2d_inv<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], tNr[1], tNc[1], Nr, Nc);

    return 0;
}



/// ----------------------------------------------------------------------------
/// ------------------------ 1D HAAR TRANSFORM ---------------------------------
/// ----------------------------------------------------------------------------


#define ONE_SQRT2 0.70710678118654746


// must be run with grid size = (Nc/2, Nr)  where Nr = numrows of input
__global__ void kern_haar1d_fwd(DTYPE* img, DTYPE* c_a, DTYPE* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nc_is_odd = (Nc & 1);
    int Nc2 = (Nc + Nc_is_odd)/2;

    if (gidy < Nr && gidx < Nc2) {
        int gidx_next = 2*gidx+1;
        if ((Nc_is_odd) && (gidx_next == Nc)) gidx_next--; // for odd size: repeat last element
        DTYPE a = img[gidy*Nc + (gidx*2)];
        DTYPE b = img[gidy*Nc + gidx_next];
        c_a[gidy* Nc2 + gidx] = ONE_SQRT2 * HAAR_AVG(a, b);
        c_d[gidy* Nc2 + gidx] = ONE_SQRT2 * HAAR_DIF(a, b);
    }
}

// must be run with grid size = (Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar1d_inv(DTYPE* img, DTYPE* c_a, DTYPE* c_d, int Nr, int Nc, int Nc2) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr && gidx < Nc2) {
        DTYPE a = c_a[gidy*Nc + (gidx/2)];
        DTYPE b = c_d[gidy*Nc + (gidx/2)];
        DTYPE res = 0.0f;
        if ((gidx & 1) == 0) res = ONE_SQRT2 * HAAR_AVG(a, b);
        else res = ONE_SQRT2 * HAAR_DIF(a, b);
        img[gidy*Nc2 + gidx] = res;
    }
}



int haar_forward1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels;
    int Nc2 = Nc;
    int Nc2_old = Nc2;
    w_div2(&Nc2);
    int tpb = 16; // TODO : tune for max perfs.
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    kern_haar1d_fwd<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], Nr, Nc);

    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2;
        w_div2(&Nc2);
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr, tpb), 1);
        kern_haar1d_fwd<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[i+1], Nr, Nc2_old);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp1, Nr*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    return 0;
}

// FIXME: for some reason, the precision of the inverse(forward) for HAAR 1D
// is not as good as in 2D
// (I have 1e-13 error for [0, 255] range in 2D, and 1e-5 in 1D)
int haar_inverse1d(DTYPE* d_image, DTYPE** d_coeffs, DTYPE* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels;
    // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNc[i] = tNc[i-1];
        w_div2(tNc + i);
    }

    int tpb = 16; // TODO : tune for max perfs.
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    for (int i = levels-1; i >= 1; i--) {
        n_blocks = dim3(w_iDivUp(tNc[i], tpb), w_iDivUp(Nr, tpb), 1);
        kern_haar1d_inv<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[i+1], Nr, tNc[i+1], tNc[i]);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0)) cudaMemcpy(d_coeffs[0], d_tmp1, Nr*tNc[1]*sizeof(DTYPE), cudaMemcpyDeviceToDevice);

    // First level
    n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    kern_haar1d_inv<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], Nr, tNc[1], Nc);

    return 0;
}



