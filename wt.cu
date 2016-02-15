/**
 *
 * TODO :
 *
 *  - W.coeffs[0] is over-written after an inversion.               (FIXME)
 *      so if  W.inverse() is run twice : invalid inverse.
 *      => Implement W.state ?
 *  - Allow both separable and non-separable without re-compiling   (OK)
 *  - User can choose the target device                             (TODO)
 *  - User can provide non-separable filters                        (TODO)
 *  - Doc ! (get_coeffs, ...)
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <cuComplex.h>
#include <time.h>

#include "wt.h"
#include "separable.cu"
#include "nonseparable.cu"
#include "haar.cu"
#include "io.h"

#  define CUDACHECK \
  { cudaThreadSynchronize(); \
    cudaError_t last = cudaGetLastError();\
    if(last!=cudaSuccess) {\
      printf("ERRORX: %s  %s  %i \n", cudaGetErrorString( last),    __FILE__, __LINE__    );    \
      exit(1);\
    }\
  }



/// ****************************************************************************
/// ******************** Wavelets class ****************************************
/// ****************************************************************************


/// Constructor : copy assignment
Wavelets& Wavelets::operator=(const Wavelets &rhs) {
  if (this != &rhs) { // protect against invalid self-assignment
    // allocate new memory and copy the elements
    size_t sz = rhs.Nr * rhs.Nc * sizeof(float);
    float* new_image, *new_tmp;
    float** new_coeffs;
    cudaMalloc(&new_image, sz);
    cudaMemcpy(new_image, rhs.d_image, sz, cudaMemcpyDeviceToDevice);

    new_coeffs =  w_create_coeffs_buffer(rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);
    w_copy_coeffs_buffer(new_coeffs, rhs.d_coeffs, rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);

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




/// Constructor : default
Wavelets::Wavelets(void) : d_image(NULL), Nr(0), Nc(0), nlevels(1), d_coeffs(NULL), do_cycle_spinning(0), d_tmp(NULL), current_shift_r(0), current_shift_c(0), do_swt(0), do_separable(1)
{
}


/// Constructor :  Wavelets from image
Wavelets::Wavelets(
    float* img,
    int Nr,
    int Nc,
    const char* wname,
    int levels,
    int memisonhost,
    int do_separable,
    int do_cycle_spinning,
    int do_swt) :

    d_image(NULL),
    Nr(Nr),
    Nc(Nc),
    nlevels(levels),
    d_coeffs(NULL),
    do_cycle_spinning(do_cycle_spinning),
    d_tmp(NULL),
    current_shift_r(0),
    current_shift_c(0),
    do_swt(do_swt),
    do_separable(do_separable),
    state(W_INIT)
{

    if (nlevels < 1) {
        puts("Warning: cannot initialize wavelet coefficients with nlevels < 1. Forcing nlevels = 1");
        nlevels = 1;
    }
      // Image
    float* d_arr_in;
    cudaMalloc(&d_arr_in, Nr*Nc*sizeof(float));
    if (!img) cudaMemset(d_arr_in, 0, Nr*Nc*sizeof(float));
    else {
        cudaMemcpyKind transfer;
        if (memisonhost) transfer = cudaMemcpyHostToDevice;
        else transfer = cudaMemcpyDeviceToDevice;
        cudaMemcpy(d_arr_in, img, Nr*Nc*sizeof(float), transfer);
    }
    this->d_image = d_arr_in;

    float* d_tmp_new;
    cudaMalloc(&d_tmp_new, 2*Nr*Nc*sizeof(float)); // Two temp. images
    this->d_tmp = d_tmp_new;

    // Coeffs
    float** d_coeffs_new;
    d_coeffs_new = w_create_coeffs_buffer(Nr, Nc, nlevels, do_swt);
    this->d_coeffs = d_coeffs_new;

    // Filters
    strncpy(this->wname, wname, 128);
    int hlen = 0;
    if (do_separable) hlen = w_compute_filters_separable(wname, do_swt);
    else hlen = w_compute_filters(wname, 1, do_swt);
    if (hlen == 0) {
        printf("ERROR: unknown wavelet name %s\n", wname);
        exit(1); // FIXME : more graceful error
    }
    this->hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N = min(Nr, Nc);
    int wmaxlev = w_ilog2(N/hlen);
    if (levels > wmaxlev) {
        printf("Warn: required level (%d) is greater than the maximum possible level for %s (%d).\n", nlevels, wname, wmaxlev);
        printf("Forcing nlevels = %d\n", wmaxlev);
        nlevels = wmaxlev;
    }
    if (do_cycle_spinning && do_swt) puts("Warning: makes little sense to use Cycle spinning with stationary Wavelet transform");
}


/// Destructor
Wavelets::~Wavelets(void) {
    if (d_image) cudaFree(d_image);
    if (d_coeffs) w_free_coeffs_buffer(d_coeffs, nlevels);
    if (d_tmp) cudaFree(d_tmp);
}

/// Method : forward
void Wavelets::forward(void) {
    if (do_cycle_spinning) {
        current_shift_r = rand() % Nr;
        current_shift_c = rand() % Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    if ((hlen == 2) && (!do_swt)) haar_forward2d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels);
    else {
        if (do_separable) {
            if (!do_swt) w_forward_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_forward_swt_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
        else {
            if (!do_swt) w_forward(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_forward_swt(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
    }
    state = W_FORWARD;
}
/// Method : inverse
void Wavelets::inverse(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if ((hlen == 2) && (!do_swt)) haar_inverse2d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels);
    else {
        if (do_separable) {
            if (!do_swt) w_inverse_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_inverse_swt_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
        else {
            w_compute_filters(wname, -1, do_swt); // TODO : dedicated inverse coeffs to avoid this computation ?
            if (!do_swt) w_inverse(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_inverse_swt(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
        if (do_cycle_spinning) {
            circshift(-current_shift_r, -current_shift_c, 1);
        }
    }
    state = W_INVERSE;
}

/// Method : soft thresholding (L1 proximal)
void Wavelets::soft_threshold(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_soft_thresh(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs);
}

/// Method : hard thresholding
void Wavelets::hard_threshold(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_hard_thresh(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs);
}

/// Method : shrink (L2 proximal)
void Wavelets::shrink(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_shrink(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs);
}



/// Method : circular shift
// If inplace = 1, the result is in d_image ; otherwise result is in d_tmp.
void Wavelets::circshift(int sr, int sc, int inplace) {
    w_call_circshift(d_image, d_tmp, Nr, Nc, sr, sc, inplace);
}
/// Method : squared L2 norm
float Wavelets::norm2sq(void) {
    float res = 0.0f;
    int Nr2, Nc2;
    if (!do_swt) { Nr2 = Nr/2; Nc2 = Nc/2; }
    else { Nr2 = Nr; Nc2 = Nc; }
    for (int i = 0; i < nlevels; i++) {
        res += cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+1], 1);
        res += cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+2], 1);
        res += cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+3], 1);
        if (!do_swt) { Nr2 /= 2; Nc2 /= 2; }
    }
    int nels = ((do_swt) ? (Nr2*Nc2) : (Nr2*Nc2*4));
    res += cublasSnrm2(nels, d_coeffs[0], 1);
    return res;
}

/// Method : L1 norm
float Wavelets::norm1(void) {
    float res = 0.0f;
    int Nr2, Nc2;
    if (!do_swt) { Nr2 = Nr/2; Nc2 = Nc/2; }
    else { Nr2 = Nr; Nc2 = Nc; }
    for (int i = 0; i < nlevels; i++) {
        res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+1], 1);
        res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+2], 1);
        res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+3], 1);
        if (!do_swt) { Nr2 /= 2; Nc2 /= 2; }
    }
    int nels = ((do_swt) ? (Nr2*Nc2) : (Nr2*Nc2*4));
    res += cublasSasum(nels, d_coeffs[0], 1);
    return res;
}

/// Method : get the image from device
int Wavelets::get_image(float* res) { // TODO: more defensive
    cudaMemcpy(res, d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
    return Nr*Nc;
}

/// Method : set the class image
void Wavelets::set_image(float* img, int mem_is_on_device) { // There are no memory check !
    cudaMemcpyKind copykind;
    if (mem_is_on_device) copykind = cudaMemcpyDeviceToDevice;
    else copykind = cudaMemcpyHostToDevice;
    cudaMemcpy(d_image, img, Nr*Nc*sizeof(float), copykind);
    state = W_INIT;
}


/// Method : get a coefficient vector from device
int Wavelets::get_coeff(float* coeff, int num) {
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        int factor = ((num == 0) ? (w_ipow2(nlevels)) : (w_ipow2((num-1)/3 +1)));
        Nr2 /= factor;
        Nc2 /= factor;
    }
    //~ printf("Retrieving %d (%d x %d)\n", num, Nr2, Nc2);
    cudaMemcpy(coeff, d_coeffs[num], Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToHost);
    return Nr2*Nc2;
}



/// Method : give some informations on the wavelet
void Wavelets::print_informations() {

    const char* state[2] = {"no", "yes"};
    puts("------------- Wavelet transform infos ------------");
    printf("Wavelet name : %s\n", wname);
    printf("Number of levels : %d\n", nlevels);
    printf("Stationary WT : %s\n", state[do_swt]);
    printf("Cycle spinning : %s\n", state[do_cycle_spinning]);
    printf("Separable transform : %s\n", state[do_separable]);

    size_t mem_used = 0;
    if (!do_swt) {
        // DWT : size(output) = size(input), since sizes are halved at each level.
        // d_image (1), d_coeffs (1), d_tmp (2)
        mem_used = 5*Nr*Nc*sizeof(float);
    }
    else {
        // SWT : size(output) = size(input)*4*levels
        // d_image (1), d_coeffs (3*levels+1), d_tmp (2)
        mem_used = (3*nlevels+4)*Nr*Nc*sizeof(float);
    }
    printf("Estimated memory footprint : %.1f MB\n", mem_used/1e6);


    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    char* device_name = strdup(properties.name);
    printf("Running on device : %s\n", device_name);
    free(device_name);
    puts("--------------------------------------------------");
}




