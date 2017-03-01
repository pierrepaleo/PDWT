#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <cuComplex.h>

//~ #include "utils.h"
#include "common.h"
#include "wt.h"
#include "separable.h"
#include "nonseparable.h"
#include "haar.h"

#  define CUDACHECK \
  { cudaThreadSynchronize(); \
    cudaError_t last = cudaGetLastError();\
    if(last!=cudaSuccess) {\
      printf("ERRORX: %s  %s  %i \n", cudaGetErrorString( last),    __FILE__, __LINE__    );    \
      exit(1);\
    }\
  }


// FIXME: temp. workaround
#define MAX_FILTER_WIDTH 40



/// ****************************************************************************
/// ******************** Wavelets class ****************************************
/// ****************************************************************************


/// Constructor : copy assignment
// do not use !
/*
Wavelets& Wavelets::operator=(const Wavelets &rhs) {
  if (this != &rhs) { // protect against invalid self-assignment
    // allocate new memory and copy the elements
    size_t sz = rhs.Nr * rhs.Nc * sizeof(DTYPE);
    DTYPE* new_image, *new_tmp;
    DTYPE** new_coeffs;
    cudaMalloc(&new_image, sz);
    cudaMemcpy(new_image, rhs.d_image, sz, cudaMemcpyDeviceToDevice);

    new_coeffs =  w_create_coeffs_buffer(rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);
    if (ndim == 2) w_copy_coeffs_buffer(new_coeffs, rhs.d_coeffs, rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);
    else  w_copy_coeffs_buffer_1d(new_coeffs, rhs.d_coeffs, rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);

    cudaMalloc(&new_tmp, sz);
    cudaMemcpy(new_tmp, rhs.d_tmp, 2*sz, cudaMemcpyDeviceToDevice); // Two temp. images

    // deallocate old memory
    cudaFree(d_image);
    w_free_coeffs_buffer(d_coeffs, nlevels);
    cudaFree(d_tmp);
    // assign the new memory to the object
    d_image = new_image;
    d_coeffs = new_coeffs;
    d_tmp = new_tmp;
    Nr = rhs.Nr;
    Nc = rhs.Nc;
    strncpy(wname, rhs.wname, 128);
    nlevels = rhs.nlevels;
    do_cycle_spinning = rhs.do_cycle_spinning;
    current_shift_r = rhs.current_shift_r;
    current_shift_c = rhs.current_shift_c;
    do_swt = rhs.do_swt;
    do_separable = rhs.do_separable;
  }
  return *this;
}
*/



/// Constructor : default
Wavelets::Wavelets(void) : d_image(NULL), d_coeffs(NULL), do_cycle_spinning(0), d_tmp(NULL), current_shift_r(0), current_shift_c(0), do_separable(1)
{
}


/// Constructor :  Wavelets from image
Wavelets::Wavelets(
    DTYPE* img,
    int Nr,
    int Nc,
    const char* wname,
    int levels,
    int memisonhost,
    int do_separable,
    int do_cycle_spinning,
    int do_swt,
    int ndim) :

    d_image(NULL),
    d_coeffs(NULL),
    do_cycle_spinning(do_cycle_spinning),
    d_tmp(NULL),
    current_shift_r(0),
    current_shift_c(0),
    do_separable(do_separable),
    state(W_INIT)
{
    winfos.Nr = Nr;
    winfos.Nc = Nc;
    winfos.nlevels = levels;
    winfos.do_swt = do_swt;
    winfos.ndims = ndim;

    if (levels < 1) {
        puts("Warning: cannot initialize wavelet coefficients with nlevels < 1. Forcing nlevels = 1");
        winfos.nlevels = 1;
    }

    // Image
    DTYPE* d_arr_in;
    cudaMalloc(&d_arr_in, Nr*Nc*sizeof(DTYPE));
    if (!img) cudaMemset(d_arr_in, 0, Nr*Nc*sizeof(DTYPE));
    else {
        cudaMemcpyKind transfer;
        if (memisonhost) transfer = cudaMemcpyHostToDevice;
        else transfer = cudaMemcpyDeviceToDevice;
        cudaMemcpy(d_arr_in, img, Nr*Nc*sizeof(DTYPE), transfer);
    }
    d_image = d_arr_in;

    DTYPE* d_tmp_new;
    cudaMalloc(&d_tmp_new, 2*Nr*Nc*sizeof(DTYPE)); // Two temp. images
    d_tmp = d_tmp_new;

    // Dimensions
    if (Nr == 1) { // 1D data
        ndim = 1;
        winfos.ndims = 1;
    }

    if (ndim == 1 && do_separable == 0) {
        puts("Warning: 1D DWT was requestred, which is incompatible with non-separable transform.");
        puts("Ignoring the do_separable option.");
        do_separable = 1;
    }
    // Filters
    strncpy(this->wname, wname, 128);
    int hlen = 0;
    if (do_separable) hlen = w_compute_filters_separable(wname, do_swt);
    else hlen = w_compute_filters(wname, 1, do_swt);
    if (hlen == 0) {
        printf("ERROR: unknown wavelet name %s\n", wname);
        //~ exit(1);
        state = W_CREATION_ERROR;
    }
    winfos.hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N;
    if (ndim == 2) N = min(Nr, Nc);
    else N = Nc;
    int wmaxlev = w_ilog2(N/hlen);
    // TODO: remove this limitation
    if (levels > wmaxlev) {
        printf("Warning: required level (%d) is greater than the maximum possible level for %s (%d) on a %dx%d image.\n", winfos.nlevels, wname, wmaxlev, winfos.Nc, winfos.Nr);
        printf("Forcing nlevels = %d\n", wmaxlev);
        winfos.nlevels = wmaxlev;
    }
    // Allocate coeffs
    DTYPE** d_coeffs_new;
    if (ndim == 1) d_coeffs_new = w_create_coeffs_buffer_1d(winfos);
    else if (ndim == 2) d_coeffs_new = w_create_coeffs_buffer(winfos);
    else {
        printf("ERROR: ndim=%d is not implemented\n", ndim);
        //~ exit(1);
        //~ throw std::runtime_error("Error on ndim");
        state = W_CREATION_ERROR;
    }
    d_coeffs = d_coeffs_new;
    if (do_cycle_spinning && do_swt) puts("Warning: makes little sense to use Cycle spinning with stationary Wavelet transform");
    // TODO
    if (do_cycle_spinning && ndim == 1) {
        puts("ERROR: cycle spinning is not implemented for 1D. Use SWT instead.");
        //~ exit(1);
        state = W_CREATION_ERROR;
    }

}




/// Constructor: copy
Wavelets::Wavelets(const Wavelets &W) :
    do_cycle_spinning(W.do_cycle_spinning),
    current_shift_c(W.current_shift_c),
    current_shift_r(W.current_shift_r),
    do_separable(W.do_separable),
    state(W.state)
{
    winfos.Nr = W.winfos.Nr;
    winfos.Nc = W.winfos.Nc;
    winfos.nlevels = W.winfos.nlevels;
    winfos.ndims = W.winfos.ndims;
    winfos.hlen = W.winfos.hlen;
    winfos.do_swt = W.winfos.do_swt;

    strncpy(wname, W.wname, 128);
    cudaMalloc(&d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE));
    cudaMemcpy(d_image, W.d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp, 2*winfos.Nr*winfos.Nc*sizeof(DTYPE));

    if (winfos.ndims == 1) {
        d_coeffs = w_create_coeffs_buffer_1d(winfos);
        w_copy_coeffs_buffer_1d(d_coeffs, W.d_coeffs, winfos);
    }
    else if (winfos.ndims == 2) {
        d_coeffs = w_create_coeffs_buffer(winfos);
        w_copy_coeffs_buffer(d_coeffs, W.d_coeffs, winfos);
    }
    else {
        puts("ERROR: 3D wavelets not implemented yet");
        state = W_CREATION_ERROR;
    }
}


/// Destructor
Wavelets::~Wavelets(void) {
    if (d_image) cudaFree(d_image);
    if (d_coeffs) {
        if (winfos.ndims == 2) w_free_coeffs_buffer(d_coeffs, winfos.nlevels);
        else w_free_coeffs_buffer_1d(d_coeffs, winfos.nlevels);
    }
    if (d_tmp) cudaFree(d_tmp);
}

/// Method : forward
void Wavelets::forward(void) {
    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    if (winfos.ndims == 1) {
        if ((winfos.hlen == 2) && (!winfos.do_swt)) haar_forward1d(d_image, d_coeffs, d_tmp, winfos);
        else {
            if (!winfos.do_swt) w_forward_separable_1d(d_image, d_coeffs, d_tmp, winfos);
            else w_forward_swt_separable_1d(d_image, d_coeffs, d_tmp, winfos);
        }
    }
    else if (winfos.ndims == 2) {
        if ((winfos.hlen == 2) && (!winfos.do_swt)) haar_forward2d(d_image, d_coeffs, d_tmp, winfos);
        else {
            if (do_separable) {
                if (!winfos.do_swt) w_forward_separable(d_image, d_coeffs, d_tmp, winfos);
                else w_forward_swt_separable(d_image, d_coeffs, d_tmp, winfos);
            }
            else {
                if (!winfos.do_swt) w_forward(d_image, d_coeffs, d_tmp, winfos);
                else w_forward_swt(d_image, d_coeffs, d_tmp, winfos);
            }
        }
    }

    // else: not implemented yet
    state = W_FORWARD;

}
/// Method : inverse
void Wavelets::inverse(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
    // TODO: handle W_INVERSE_ERROR with return codes of inverse transforms
    if (winfos.ndims == 1) {
        if ((winfos.hlen == 2) && (!winfos.do_swt)) haar_inverse1d(d_image, d_coeffs, d_tmp, winfos);
        else {
            if (!winfos.do_swt) w_inverse_separable_1d(d_image, d_coeffs, d_tmp, winfos);
            else w_inverse_swt_separable_1d(d_image, d_coeffs, d_tmp, winfos);
        }
    }
    else if (winfos.ndims == 2) {
        if ((winfos.hlen == 2) && (!winfos.do_swt)) haar_inverse2d(d_image, d_coeffs, d_tmp, winfos);
        else {
            if (do_separable) {
                if (!winfos.do_swt) w_inverse_separable(d_image, d_coeffs, d_tmp, winfos);
                else w_inverse_swt_separable(d_image, d_coeffs, d_tmp, winfos);
            }
            else {
                w_compute_filters(wname, -1, winfos.do_swt); // TODO : dedicated inverse coeffs to avoid this computation ?
                if (!winfos.do_swt) w_inverse(d_image, d_coeffs, d_tmp, winfos);
                else w_inverse_swt(d_image, d_coeffs, d_tmp, winfos);
            }
        }
    }
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}

/// Method : soft thresholding (L1 proximal)
void Wavelets::soft_threshold(DTYPE beta, int do_thresh_appcoeffs, int normalize) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_soft_thresh(d_coeffs, beta, winfos, do_thresh_appcoeffs, normalize);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}

/// Method : hard thresholding
void Wavelets::hard_threshold(DTYPE beta, int do_thresh_appcoeffs, int normalize) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_hard_thresh(d_coeffs, beta, winfos, do_thresh_appcoeffs, normalize);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}

/// Method : shrink (L2 proximal)
void Wavelets::shrink(DTYPE beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_shrink(d_coeffs, beta, winfos, do_thresh_appcoeffs);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}
/// Method : projection onto the L-infinity ball (infinity norm proximal, i.e dual L1 norm proximal)
void Wavelets::proj_linf(DTYPE beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_proj_linf(d_coeffs, beta, winfos, do_thresh_appcoeffs);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}





/// Method : circular shift
// If inplace = 1, the result is in d_image ; otherwise result is in d_tmp.
void Wavelets::circshift(int sr, int sc, int inplace) {
    w_call_circshift(d_image, d_tmp, winfos, sr, sc, inplace);
}
/// Method : squared L2 norm
DTYPE Wavelets::norm2sq(void) {
    DTYPE res = 0.0f;
    int Nr2 = winfos.Nr;
    int Nc2 = winfos.Nc;
    DTYPE tmp = 0;
    for (int i = 0; i < winfos.nlevels; i++) {
        if (!winfos.do_swt) {
            if (winfos.ndims > 1) w_div2(&Nr2);
            w_div2(&Nc2);
        }
        if (winfos.ndims == 2) { // 2D
            tmp = cublas_nrm2(Nr2*Nc2, d_coeffs[3*i+1], 1);
            res += tmp*tmp;
            tmp =cublas_nrm2(Nr2*Nc2, d_coeffs[3*i+2], 1);
            res += tmp*tmp;
            tmp = cublas_nrm2(Nr2*Nc2, d_coeffs[3*i+3], 1);
            res += tmp*tmp;
        }
        else { // 1D
            res += cublas_asum(Nr2*Nc2, d_coeffs[i+1], 1);
        }
    }
    tmp = cublas_nrm2(Nr2*Nc2, d_coeffs[0], 1);
    res += tmp*tmp;
    return res;
}

/// Method : L1 norm
DTYPE Wavelets::norm1(void) {
    DTYPE res = 0.0f;
    int Nr2 = winfos.Nr;
    int Nc2 = winfos.Nc;
    for (int i = 0; i < winfos.nlevels; i++) {
        if (!winfos.do_swt) {
            if (winfos.ndims > 1) w_div2(&Nr2);
            w_div2(&Nc2);
        }
        if (winfos.ndims == 2) { // 2D
            res += cublas_asum(Nr2*Nc2, d_coeffs[3*i+1], 1);
            res += cublas_asum(Nr2*Nc2, d_coeffs[3*i+2], 1);
            res += cublas_asum(Nr2*Nc2, d_coeffs[3*i+3], 1);
        }
        else { // 1D
            res += cublas_asum(Nr2*Nc2, d_coeffs[i+1], 1);
        }
    }
    res += cublas_asum(Nr2*Nc2, d_coeffs[0], 1);
    return res;
}

/// Method : get the image from device
int Wavelets::get_image(DTYPE* res) { // TODO: more defensive
    cudaMemcpy(res, d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToHost);
    return winfos.Nr*winfos.Nc;
}

/// Method : set the class image
void Wavelets::set_image(DTYPE* img, int mem_is_on_device) { // There are no memory check !
    cudaMemcpyKind copykind;
    if (mem_is_on_device) copykind = cudaMemcpyDeviceToDevice;
    else copykind = cudaMemcpyHostToDevice;
    cudaMemcpy(d_image, img, winfos.Nr*winfos.Nc*sizeof(DTYPE), copykind);
    state = W_INIT;
}


/// Method : set a coefficient
void Wavelets::set_coeff(DTYPE* coeff, int num, int mem_is_on_device) { // There are no memory check !
    cudaMemcpyKind copykind;
    if (mem_is_on_device) copykind = cudaMemcpyDeviceToDevice;
    else copykind = cudaMemcpyHostToDevice;
    int Nr2 = winfos.Nr, Nc2 = winfos.Nc;
    if (winfos.ndims == 2) {
        // In 2D, num stands for the following:
        // A  H1 V1 D1  H2 V2 D2
        // 0  1  2  3   4  5  6
        // for num>0,  1+(num-1)/3 tells the scale number
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = ((num-1)/3) +1;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nr2);
            w_div2(&Nc2);
        }
    }
    else if (winfos.ndims == 1) {
        // In 1D, num stands for the following:
        // A  D1 D2 D3
        // 0  1  2  3
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = num;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nc2);
        }
    }
    cudaMemcpy(d_coeffs[num], coeff, Nr2*Nc2*sizeof(DTYPE), copykind);
    //~ state = W_FORWARD; // ?
}





/// Method : get a coefficient vector from device
int Wavelets::get_coeff(DTYPE* coeff, int num) {
    if (state == W_INVERSE) {
        puts("Warning: get_coeff(): inverse() has been performed, the coefficients has been modified and do not make sense anymore.");
        return 0;
    }
    int Nr2 = winfos.Nr, Nc2 = winfos.Nc;
    if (winfos.ndims == 2) {
        // In 2D, num stands for the following:
        // A  H1 V1 D1  H2 V2 D2
        // 0  1  2  3   4  5  6
        // for num>0,  1+(num-1)/3 tells the scale number
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = ((num-1)/3) +1;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nr2);
            w_div2(&Nc2);
        }
    }
    else if (winfos.ndims == 1) {
        // In 1D, num stands for the following:
        // A  D1 D2 D3
        // 0  1  2  3
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = num;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nc2);
        }
    }
    //~ printf("Retrieving %d (%d x %d)\n", num, Nr2, Nc2);
    cudaMemcpy(coeff, d_coeffs[num], Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToHost); //TODO: handle DeviceToDevice ?
    return Nr2*Nc2;
}



/// Method : give some informations on the wavelet
void Wavelets::print_informations() {

    const char* state[2] = {"no", "yes"};
    puts("------------- Wavelet transform infos ------------");
    printf("Data dimensions : ");
    if (winfos.ndims == 2) printf("(%d, %d)\n", winfos.Nr, winfos.Nc);
    else { // 1D
        if (winfos.Nr == 1) printf("%d\n", winfos.Nc);
        else printf("(%d, %d) [batched 1D transform]\n", winfos.Nr, winfos.Nc);
    }
    printf("Wavelet name : %s\n", wname);
    printf("Number of levels : %d\n", winfos.nlevels);
    printf("Stationary WT : %s\n", state[winfos.do_swt]);
    printf("Cycle spinning : %s\n", state[do_cycle_spinning]);
    printf("Separable transform : %s\n", state[do_separable]);

    size_t mem_used = 0;
    if (!winfos.do_swt) {
        // DWT : size(output) = size(input), since sizes are halved at each level.
        // d_image (1), d_coeffs (1), d_tmp (2)
        mem_used = 5*winfos.Nr*winfos.Nc*sizeof(DTYPE);
    }
    else {
        // SWT : size(output) = size(input)*4*levels
        // d_image (1), d_coeffs (3*levels+1), d_tmp (2)
        if (winfos.ndims == 2) mem_used = (3*winfos.nlevels+4)*winfos.Nr*winfos.Nc*sizeof(DTYPE);
        else mem_used = (winfos.nlevels+4)*winfos.Nr*winfos.Nc*sizeof(DTYPE);
    }
    printf("Estimated memory footprint : %.2f MB\n", mem_used/1e6);


    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    char* device_name = strdup(properties.name);
    printf("Running on device : %s\n", device_name);
    free(device_name);
    puts("--------------------------------------------------");
}


/// Provide a custom filter bank to the current Wavelet instance.
/// If do_separable = 1, the filters are expected to be L, H.
/// Otherwise, the filters are expected to be A, H, V, D (square size)
int Wavelets::set_filters_forward(char* filtername, uint len, DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4) {
    if (len > MAX_FILTER_WIDTH) {
        printf("ERROR: Wavelets.set_filters_forward(): filter length (%d) exceeds the maximum size (%d)\n", len, MAX_FILTER_WIDTH);
        return -1;
    }
    if (do_separable) {
        if (cudaMemcpyToSymbol(c_kern_L, filter1, len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_H, filter2, len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            return -3;
        }
    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_forward(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        if (cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            return -3;
        }
    }
    winfos.hlen = len;
    strncpy(wname, filtername, 128);

    return 0;
}

/// Here the filters are assumed to be of the same size of those provided to set_filters_forward()
int Wavelets::set_filters_inverse(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4) {
    uint len = winfos.hlen;
    if (do_separable) {
        // ignoring args 4 and 5
        if (cudaMemcpyToSymbol(c_kern_IL, filter1, len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_IH, filter2, len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            return -3;
        }
    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_inverse(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        // The same symbols are used for the inverse filters
        if (cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            return -3;
        }
    }

    return 0;
}




/// ----------------------------------------------------------------------------
/// --------- Operators... for now I am not considering overloading  -----------
/// ----------------------------------------------------------------------------


/**
 * \brief In-place addition of wavelets coefficients
 *
 *  For a given instance "Wav" of the class Wavelets, it performs
 *   Wav += W. Only the wavelets coefficients are added, the image attribute
 *  is not replaced
 *
 *
 * \param W : Wavelets class instance
 * \return 0 if no error
 *
 */
int Wavelets::add_wavelet(Wavelets W, DTYPE alpha) {

    // Various checks
    if ((winfos.nlevels != W.winfos.nlevels) || (strcasecmp(wname, W.wname))) {
        puts("ERROR: add_wavelet(): right operand is not the same transform (wname, level)");
        return -1;
    }
    if (state == W_INVERSE || W.state == W_INVERSE) {
        puts("WARNING: add_wavelet(): this operation makes no sense when wavelet has just been inverted");
        return 1;
    }
    if (winfos.Nr != W.winfos.Nr || winfos.Nc != W.winfos.Nc || winfos.ndims != W.winfos.ndims) {
        puts("ERROR: add_wavelet(): operands do not have the same geometry");
        return -2;
    }
    if ((winfos.do_swt) ^ (W.winfos.do_swt)) {
        puts("ERROR: add_wavelet(): operands should both use SWT or DWT");
        return -3;
    }
    if (
        (do_cycle_spinning * W.do_cycle_spinning)
        && (
            (current_shift_r != W.current_shift_r) || (current_shift_c != W.current_shift_c)
           )
       )
    {
        puts("ERROR: add_wavelet(): operands do not have the same current shift");
        return -4;
    }

    if (winfos.ndims == 1) w_add_coeffs_1d(d_coeffs, W.d_coeffs, winfos, alpha);
    else w_add_coeffs(d_coeffs, W.d_coeffs, winfos, alpha);
    return 0;
}













