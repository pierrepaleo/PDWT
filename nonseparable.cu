#include "nonseparable.h"
#include "common.h"

// outer product of arrays "a", "b" of length "len"
float* w_outer(float* a, float* b, int len) {
    float* res = (float*) calloc(len*len, sizeof(float));
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            res[i*len+j] = a[i]*b[j];
        }
    }
    return res;
}


/// Compute the four filters A, H, V, D  from a family name.
/// These filters are separable, i.e computed from 1D filters.
/// wname: name of the filter ("haar", "db3", "sym4", ...)
/// direction: 1 for forward transform, -1 for inverse transform
/// Returns : the filter width "hlen" if success ; a negative value otherwise.
int w_compute_filters(const char* wname, int direction, int do_swt) {
    if (direction == 0) {
        puts("ERROR: w_compute_filters(): please specify a direction for second argument : +1 for forward, -1 for inverse)");
        return -1;
    }
    int hlen = 0;
    float* f1_l; // 1D lowpass
    float* f1_h; // 1D highpass
    float* f2_a, *f2_h, *f2_v, *f2_d; // 2D filters

    // Haar filters has specific kernels
    if (!do_swt) {
        if ((!strcasecmp(wname, "haar")) || (!strcasecmp(wname, "db1")) || (!strcasecmp(wname, "bior1.1")) || (!strcasecmp(wname, "rbior1.1"))) {
            return 2;
        }
    }

    // Browse available filters (see filters.h)
    int i;
    for (i = 0; i < 72; i++) {
        if (!strcasecmp(wname, all_filters[i].wname)) {
            hlen = all_filters[i].hlen;
            if (direction > 0) {
                f1_l = all_filters[i].f_l;
                f1_h = all_filters[i].f_h;
            }
            else {
                f1_l = all_filters[i].i_l;
                f1_h = all_filters[i].i_h;
            }
            break;
        }
    }
    if (hlen == 0) {
        printf("ERROR: w_compute_filters(): unknown filter %s\n", wname);
        return -2;
    }

    // Create the separable 2D filters
    f2_a = w_outer(f1_l, f1_l, hlen);
    f2_h = w_outer(f1_l, f1_h, hlen); // CHECKME
    f2_v = w_outer(f1_h, f1_l, hlen);
    f2_d = w_outer(f1_h, f1_h, hlen);

    // Copy the filters to device constant memory
    cudaMemcpyToSymbol(c_kern_LL, f2_a, hlen*hlen*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_LH, f2_h, hlen*hlen*sizeof(float), 0, cudaMemcpyHostToDevice); // CHECKME
    cudaMemcpyToSymbol(c_kern_HL, f2_v, hlen*hlen*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_HH, f2_d, hlen*hlen*sizeof(float), 0, cudaMemcpyHostToDevice);

    return hlen;
}






// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input image
__global__ void w_kern_forward(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr2 = Nr/2, Nc2 = Nc/2;
    if (gidy < Nr2 && gidx < Nc2) {
        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jy1 = c - 2*gidy;
        int jy2 = Nr - 1 - 2*gidy + c;
        int jx1 = c - 2*gidx;
        int jx2 = Nc - 1 - 2*gidx + c;
        float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        float img_val;

        // Convolution with periodic boundaries extension.
        // The following can be sped-up by splitting into 3*3 loops, but it would be a nightmare for readability
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy*2 - c + jy;
            if (jy < jy1) idx_y += Nr;
            if (jy > jy2) idx_y -= Nr ;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx*2 - c + jx;
                if (jx < jx1) idx_x += Nc;
                if (jx > jx2) idx_x -= Nc ;

                img_val = img[idx_y*Nc + idx_x];
                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
            }
        }
        c_a[(gidy)* Nc2 + (gidx)] = res_a;
        c_h[(gidy)* Nc2 + (gidx)] = res_h;
        c_v[(gidy)* Nc2 + (gidx)] = res_v;
        c_d[(gidy)* Nc2 + (gidx)] = res_d;
    }
}




// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void w_kern_inverse(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr2 = Nr*2, Nc2 = Nc*2;
     //~ if ((gidy < Nr2-10 && gidx < Nc2-10) && (gidx > 10 && gidy > 10)) {
    if (gidy < Nr2 && gidx < Nc2) {

        int c, hL, hR;
        int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
        if (hlen2 & 1) { // odd half-kernel size
            c = hlen2/2;
            hL = c;
            hR = c;
        }
        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
            c = hlen2/2 - 0;
            hL = c;
            hR = c-1;
            // virtual id for shift
            // TODO : for the very first convolution (on the edges), this is not exactly accurate (?)
            gidx += 1;
            gidy += 1;
        }
        int jy1 = c - gidy/2;
        int jy2 = Nr - 1 - gidy/2 + c;
        int jx1 = c - gidx/2;
        int jx2 = Nc - 1 - gidx/2 + c;

        // There are 4 threads/coeff index. Each thread will do a convolution with the even/odd indices of the kernels along each dimension.
        int offset_x = 1-(gidx & 1);
        int offset_y = 1-(gidy & 1);

        float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy/2 - c + jy;
            if (jy < jy1) idx_y += Nr;
            if (jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx/2 - c + jx;
                if (jx < jx1) idx_x += Nc;
                if (jx > jx2) idx_x -= Nc;

                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
            }
        }
        if ((hlen2 & 1) == 1) img[gidy * Nc2 + gidx] = res_a + res_h + res_v + res_d;
        else img[(gidy-1) * Nc2 + (gidx-1)] = res_a + res_h + res_v + res_d;
    }
}







int w_forward(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos) {

    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int tpb = 16; // TODO : tune for max perfs.
    int Nc2 = Nc/2, Nr2 = Nr/2;
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    w_kern_forward<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr2*2, Nc2*2, hlen);

    for (int i=1; i < levels; i++) {
        Nc2 /= 2;
        Nr2 /= 2;
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_forward<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr2*2, Nc2*2, hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}


int w_inverse(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nr0 = Nr, Nc0 = Nc;
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
        w_kern_inverse<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc, hlen);
        Nr *= 2;
        Nc *= 2;
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, (Nr0/2)*(Nc0/2)*sizeof(float), cudaMemcpyDeviceToDevice);
    //~ CUDACHECK;
    // First level
    n_blocks = dim3(w_iDivUp(Nc*2, tpb), w_iDivUp(Nr*2, tpb), 1);
    w_kern_inverse<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen);

    return 0;
}





/// ----------------------------------------------------------------------------
/// -------------------------   Undecimated DWT --------------------------------
/// ----------------------------------------------------------------------------



// must be run with grid size = (Nc, Nr)  where Nr = numrows of input image
__global__ void w_kern_forward_swt(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen, int level) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr && gidx < Nc) {

        int factor = 1 << (level - 1);
        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }

        c *= factor;
        int jx1 = c - gidx;
        int jx2 = Nc - 1 - gidx + c;
        int jy1 = c - gidy;
        int jy2 = Nr - 1 - gidy + c;
        float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        float img_val;

        // Convolution with periodic boundaries extension.
        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + factor*jy;
            if (factor*jy < jy1) idx_y += Nr;
            if (factor*jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx + jx*factor - c;
                if (factor*jx < jx1) idx_x += Nc;
                if (factor*jx > jx2) idx_x -= Nc;

                img_val = img[idx_y*Nc + idx_x];
                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];

            }
        }
        c_a[gidy* Nc + gidx] = res_a;
        c_h[gidy* Nc + gidx] = res_h;
        c_v[gidy* Nc + gidx] = res_v;
        c_d[gidy* Nc + gidx] = res_d;
    }
}




// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void w_kern_inverse_swt(float* img, float* c_a, float* c_h, float* c_v, float* c_d, int Nr, int Nc, int hlen, int level) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr && gidx < Nc) {

        int factor = 1 << (level - 1);
        int c, hL, hR;
        if (hlen & 1) { // odd half-kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
            c = hlen/2 - 0;
            hL = c;
            hR = c-1;
        }
        c *= factor;
        int jy1 = c - gidy;
        int jy2 = Nr - 1 - gidy + c;
        int jx1 = c - gidx;
        int jx2 = Nc - 1 - gidx + c;

        float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + jy*factor;
            if (factor*jy < jy1) idx_y += Nr;
            if (factor*jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx - c + jx*factor;
                if (factor*jx < jx1) idx_x += Nc;
                if (factor*jx > jx2) idx_x -= Nc;

                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
            }
        }
        img[gidy * Nc + gidx] = res_a + res_h + res_v + res_d;
    }
}






int w_forward_swt(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    int tpb = 16; // TODO : tune for max perfs.
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    w_kern_forward_swt<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen, 1);
    for (int i=1; i < levels; i++) {
        w_kern_forward_swt<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}



int w_inverse_swt(float* d_image, float** d_coeffs, float* d_tmp, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    float* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    int tpb = 16; // TODO : tune for max perfs.
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    for (int i = levels-1; i >= 1; i--) {
        w_kern_inverse_swt<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
    // First scale
    n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    w_kern_inverse_swt<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen, 1);

    return 0;
}



