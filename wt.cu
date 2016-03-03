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
//~ #include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <cuComplex.h>
//~ #include <time.h>

#include "wt.h"
#include "separable.cu"
#include "nonseparable.cu"
#include "haar.cu"
//~ #include "io.h"

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
// do not use !
/*
Wavelets& Wavelets::operator=(const Wavelets &rhs) {
  if (this != &rhs) { // protect against invalid self-assignment
    // allocate new memory and copy the elements
    size_t sz = rhs.Nr * rhs.Nc * sizeof(float);
    float* new_image, *new_tmp;
    float** new_coeffs;
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
    int do_swt,
    int ndim) :

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
    state(W_INIT),
    ndim(ndim)
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

    // Dimensions
    if (Nr == 1) { // detect 1D data
        ndim = 1;
        this->ndim = 1;
    }

    // Coeffs
    float** d_coeffs_new;
    if (ndim == 1) d_coeffs_new = w_create_coeffs_buffer_1d(Nr, Nc, nlevels, do_swt);
    else if (ndim == 2) d_coeffs_new = w_create_coeffs_buffer(Nr, Nc, nlevels, do_swt);
    else {
        printf("ERROR: ndim=%d is not implemented\n", ndim);
        exit(1);
    }
    this->d_coeffs = d_coeffs_new;

    if (ndim == 1 && do_separable == 0) {
        puts("Warning: requestred 1D DWT, which is incompatible with non-separable transform.");
        puts("Forcing do_separable = 1");
        do_separable = 1;
    }
    // Filters
    strncpy(this->wname, wname, 128);
    int hlen = 0;
    if (do_separable) hlen = w_compute_filters_separable(wname, do_swt);
    else hlen = w_compute_filters(wname, 1, do_swt);
    if (hlen == 0) {
        printf("ERROR: unknown wavelet name %s\n", wname);
        exit(1);
    }
    this->hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N;
    if (ndim == 2) N = min(Nr, Nc);
    else N = Nc;
    int wmaxlev = w_ilog2(N/hlen);
    if (levels > wmaxlev) {
        printf("Warning: required level (%d) is greater than the maximum possible level for %s (%d).\n", nlevels, wname, wmaxlev);
        printf("Forcing nlevels = %d\n", wmaxlev);
        nlevels = wmaxlev;
    }
    if (do_cycle_spinning && do_swt) puts("Warning: makes little sense to use Cycle spinning with stationary Wavelet transform");
    // TODO
    if (do_cycle_spinning && ndim == 1) {
        puts("ERROR: cycle spinning is not implemented for 1D. Use SWT instead.");
        exit(1);
    }

}




/// Constructor: copy
Wavelets::Wavelets(const Wavelets &W) :
    Nr(W.Nr),
    Nc(W.Nc),
    nlevels(W.nlevels),
    do_cycle_spinning(W.do_cycle_spinning),
    current_shift_c(W.current_shift_c),
    current_shift_r(W.current_shift_r),
    hlen(W.hlen),
    do_swt(W.do_swt),
    do_separable(W.do_separable),
    state(W.state),
    ndim(W.ndim)
{

    strncpy(wname, W.wname, 128);
    cudaMalloc(&d_image, Nr*Nc*sizeof(float));
    cudaMemcpy(d_image, W.d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp, 2*Nr*Nc*sizeof(float));
    //~ cudaMemcpy(d_tmp, W.d_tmp, 2*Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice); // not required

    if (ndim == 1) {
        d_coeffs = w_create_coeffs_buffer_1d(Nr, Nc, nlevels, do_swt);
        w_copy_coeffs_buffer_1d(d_coeffs, W.d_coeffs, Nr, Nc, nlevels, do_swt);
    }
    else if (ndim == 2) {
        d_coeffs = w_create_coeffs_buffer(Nr, Nc, nlevels, do_swt);
        w_copy_coeffs_buffer(d_coeffs, W.d_coeffs, Nr, Nc, nlevels, do_swt);
    }
    else {
        puts("ERROR: 3D wavelets not implemented yet");
        exit(-1);
    }
}


/// Destructor
Wavelets::~Wavelets(void) {
    if (d_image) cudaFree(d_image);
    if (d_coeffs) {
        if (ndim == 2) w_free_coeffs_buffer(d_coeffs, nlevels);
        else w_free_coeffs_buffer_1d(d_coeffs, nlevels);
    }
    if (d_tmp) cudaFree(d_tmp);
}

/// Method : forward
void Wavelets::forward(void) {
    if (do_cycle_spinning) {
        current_shift_r = rand() % Nr;
        current_shift_c = rand() % Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    if (ndim == 1) {
        if ((hlen == 2) && (!do_swt)) haar_forward1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels);
        else {
            if (!do_swt) w_forward_separable_1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_forward_swt_separable_1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
    }
    else if (ndim == 2) {
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
    if (ndim == 1) {
        if ((hlen == 2) && (!do_swt)) haar_inverse1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels);
        else {
            if (!do_swt) w_inverse_separable_1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_inverse_swt_separable_1d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        }
    }
    else if (ndim == 2) {
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
        }
    }
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}

/// Method : soft thresholding (L1 proximal)
void Wavelets::soft_threshold(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_soft_thresh(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs, ndim);
}

/// Method : hard thresholding
void Wavelets::hard_threshold(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_hard_thresh(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs, ndim);
}

/// Method : shrink (L2 proximal)
void Wavelets::shrink(float beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_shrink(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs, ndim);
}



/// Method : circular shift
// If inplace = 1, the result is in d_image ; otherwise result is in d_tmp.
void Wavelets::circshift(int sr, int sc, int inplace) {
    w_call_circshift(d_image, d_tmp, Nr, Nc, sr, sc, inplace, ndim);
}
/// Method : squared L2 norm
float Wavelets::norm2sq(void) {
    float res = 0.0f;
    int Nr2, Nc2;
    if (!do_swt) { Nr2 = Nr/2; Nc2 = Nc/2; }
    else { Nr2 = Nr; Nc2 = Nc; }
    float tmp = 0;
    for (int i = 0; i < nlevels; i++) {
        tmp = cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+1], 1);
        res += tmp*tmp;
        tmp =cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+2], 1);
        res += tmp*tmp;
        tmp = cublasSnrm2(Nr2*Nc2, d_coeffs[3*i+3], 1);
        res += tmp*tmp;
        if (!do_swt) { Nr2 /= 2; Nc2 /= 2; }
    }
    int nels = ((do_swt) ? (Nr2*Nc2) : (Nr2*Nc2*4));
    tmp = cublasSnrm2(nels, d_coeffs[0], 1);
    res += tmp*tmp;
    return res;
}

/// Method : L1 norm
float Wavelets::norm1(void) {
    float res = 0.0f;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) { if (ndim > 1) Nr2 = Nr/2; Nc2 = Nc/2; }
    for (int i = 0; i < nlevels; i++) {
        if (ndim == 2) { // 2D
            res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+1], 1);
            res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+2], 1);
            res += cublasSasum(Nr2*Nc2, d_coeffs[3*i+3], 1);
        }
        else { // 1D
            res += cublasSasum(Nr2*Nc2, d_coeffs[i+1], 1);
        }
        if (!do_swt) { if (ndim > 1) Nr2 /= 2; Nc2 /= 2; }
    }
    int nels;
    if (ndim == 2) nels = ((do_swt) ? (Nr2*Nc2) : (Nr2*Nc2*4));
    else nels = ((do_swt) ? (Nr2*Nc2) : (Nr2*Nc2*2));
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
    if (state == W_INVERSE) {
        puts("Warning: get_coeff(): inverse() has been performed, the coefficients has been modified and do not make sense anymore.");
        // TODO : then what ?
    }
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndim == 2) {
            int factor = ((num == 0) ? (w_ipow2(nlevels)) : (w_ipow2((num-1)/3 +1)));
            Nr2 /= factor;
            Nc2 /= factor;
        }
        else { // (ndim == 1)
            int factor = ((num == 0) ? (w_ipow2(nlevels)) : (w_ipow2((num-1) +1)));
            Nc2 /= factor;
        }
    }
    //~ printf("Retrieving %d (%d x %d)\n", num, Nr2, Nc2);
    cudaMemcpy(coeff, d_coeffs[num], Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToHost);
    return Nr2*Nc2;
}



/// Method : give some informations on the wavelet
void Wavelets::print_informations() {

    const char* state[2] = {"no", "yes"};
    puts("------------- Wavelet transform infos ------------");
    printf("Data dimensions : ");
    if (ndim == 2) printf("(%d, %d)\n", Nr, Nc);
    else { // 1D
        if (Nr == 1) printf("%d\n", Nc);
        else printf("(%d, %d) [batched 1D transform]\n", Nr, Nc);
    }
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
        if (ndim == 2) mem_used = (3*nlevels+4)*Nr*Nc*sizeof(float);
        else mem_used = (nlevels+4)*Nr*Nc*sizeof(float);
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
int Wavelets::set_filters_forward(int len, float* filter1, float* filter2, float* filter3, float* filter4) {
    if (len > MAX_FILTER_WIDTH) {
        printf("ERROR: Wavelets.set_filters_forward(): filter length (%d) exceeds the maximum size (%d)\n", len, MAX_FILTER_WIDTH);
        return -1;
    }
    if (do_separable) {
        cudaMemcpyToSymbol(c_kern_L, filter1, len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_H, filter2, len*sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_forward(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    hlen = len;
    return 0;
}

/// Here the filters are assumed to be of the same size of those provided to set_filters_forward()
int Wavelets::set_filters_inverse(float* filter1, float* filter2, float* filter3, float* filter4) {
    int len = hlen;
    if (do_separable) {
        // ignoring args 4 and 5
        cudaMemcpyToSymbol(c_kern_IL, filter1, len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_IH, filter2, len*sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_inverse(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        // The same symbols are used for the inverse filters
        cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(float), 0, cudaMemcpyHostToDevice);
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
int Wavelets::add_wavelet(Wavelets W, float alpha) {

    // Various checks
    if ((this->nlevels != W.nlevels) || (strcasecmp(this->wname, W.wname))) {
        puts("ERROR: add_wavelet(): right operand is not the same transform (wname, level)");
        return -1;
    }
    if (this->state == W_INVERSE || W.state == W_INVERSE) {
        puts("WARNING: add_wavelet(): this operation makes no sense when wavelet has just been inverted");
        return 1;
    }
    if (this->Nr != W.Nr || this->Nc != W.Nc || this->ndim != W.ndim) {
        puts("ERROR: add_wavelet(): operands do not have the same geometry");
        return -2;
    }
    if ((this->do_swt) ^ (W.do_swt)) {
        puts("ERROR: add_wavelet(): operands should both use SWT or DWT");
        return -3;
    }
    if (
        (this->do_cycle_spinning * W.do_cycle_spinning)
        && (
            (this->current_shift_r != W.current_shift_r) || (this->current_shift_c != W.current_shift_c)
           )
       )
    {
        puts("ERROR: add_wavelet(): operands do not have the same current shift");
        return -4;
    }

    // -----

    if (this->ndim == 1) w_add_coeffs_1d(this->d_coeffs, W.d_coeffs, this->Nr, this->Nc, this->nlevels, this->do_swt, alpha);
    else w_add_coeffs(this->d_coeffs, W.d_coeffs, this->Nr, this->Nc, this->nlevels, this->do_swt, alpha);





    return 0;
}













