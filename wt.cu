/**
 *
 * TODO :
 *  - Better structure/separation between kernels and host calls
 *  - Allow both separable and non-separable without re-compiling
 *  - User can choose the target device
 *  - User can provide non-separable filters
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <cuComplex.h>
#include <time.h>
#include "filters.h"

#define DO_SEPARABLE 1
#define W_BLKSIZE 16


/// ****************************************************************************
/// ******************************** Macros ************************************
/// ****************************************************************************


#if DO_SEPARABLE != 0
    #include "separable.cu"
#else
    #include "nonseparable.cu"
#endif
#include "haar.cu"

#  define CUDACHECK \
  { cudaThreadSynchronize(); \
    cudaError_t last = cudaGetLastError();\
    if(last!=cudaSuccess) {\
      printf("ERRORX: %s  %s  %i \n", cudaGetErrorString( last),    __FILE__, __LINE__    );    \
      exit(1);\
    }\
  }


/// ****************************************************************************
/// ****************** I/O (simple raw read/write)  ****************************
/// ****************************************************************************



float* read_dat_file_float(const char* fname, int len) {
    FILE* fid = fopen(fname, "rb");
    if (fid == NULL) {
        printf("ERROR in read_dat_file_float(): could not read %s\n", fname);
        return NULL;
    }
    float* out = (float*) calloc(len, sizeof(float));
    fread(out, len, sizeof(float), fid);
    fclose(fid);
    return out;
}

void write_dat_file_float(const char* fname, float* arr, int len) {
    FILE* fid = fopen(fname, "wb");
    if (fid == NULL) return;
    fwrite(arr, len, sizeof(float), fid);
    fclose(fid);
}



/// ****************************************************************************
/// ******************** CUDA Kernels and calls ********************************
/// ****************************************************************************


/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
__global__ void wavelets_kern_soft_thresh(float* c_h, float* c_v, float* c_d, float beta, int Nr, int Nc) {
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

__global__ void wavelets_kern_soft_thresh_appcoeffs(float* c_a, float beta, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    float val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
}


void wavelets_call_soft_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs) {
    int tpb = 16; // TODO : tune for max perfs.
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        Nr2 /= 2;
        Nc2 /= 2;
    }
    if (do_thresh_appcoeffs) {
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        wavelets_kern_soft_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            Nr /= 2;
            Nc /= 2;
        }
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        wavelets_kern_soft_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
    }
}


__global__ void wavelets_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidx < Nc && gidy < Nr) {
        int r = gidy - sr, c = gidx - sc;
        if (r < 0) r += Nr;
        if (c < 0) c += Nc;
        d_out[gidy*Nc + gidx] = d_image[r*Nc + c];
    }
}

// if inplace = 1, the result is in "d_image" ; otherwise result is in "d_image2".
void wavelets_call_circshift(float* d_image, float* d_image2, int Nr, int Nc, int sr, int sc, int inplace = 1) {
    // Modulus in C can be negative
    if (sr < 0) sr += Nr; // or do while loops to ensure positive numbers
    if (sc < 0) sc += Nc;
    sr = sr % Nr;
    sc = sc % Nc;
    dim3 n_blocks = dim3(w_iDivUp(Nc, W_BLKSIZE), w_iDivUp(Nr, W_BLKSIZE), 1);
    dim3 n_threads_per_block = dim3(W_BLKSIZE, W_BLKSIZE, 1);
    if (inplace) {
        cudaMemcpy(d_image2, d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToDevice);
        wavelets_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image2, d_image, Nr, Nc, sr, sc);
    }
    else {
        wavelets_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image, d_image2, Nr, Nc, sr, sc);
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



/// ****************************************************************************
/// ******************** Wavelets class ****************************************
/// ****************************************************************************

class Wavelets {
  public:
    // *******
    // Members
    // *******
    int Nr; // Number of rows of the image
    int Nc; // Number of columns of the image
    int nlevels; // Number of levels for the wavelet decomposition
    float* d_image; // Image (input or result of reconstruction), on device
    float** d_coeffs; // Wavelet coefficients, on device
    float* d_tmp; // Temporary device array (to avoid multiple malloc/free)
    int do_cycle_spinning;
    int current_shift_r;
    int current_shift_c;
    int hlen; // Filter length
    char wname[128]; // Wavelet name
    int do_swt; // 1 if performing undecimated WT



    // ********
    // Basics
    // ********
    // Default constructor
    Wavelets();
    // Constructor : Wavelets from image
    Wavelets(float* img, int Nr, int Nc, const char* wname, int levels, int memisonhost=0, int do_cycle_spinning = 0, int do_swt = 0);
    // Constructor : Wavelets from coeffs
    Wavelets(float** d_thecoeffs, int Nr, int Nc, const char* wname, int levels, int do_cycle_spinning);
    // Class copy (constructor)
    Wavelets(const Wavelets &W);  // Pass by non-const reference ONLY if the function will modify the parameter and it is the intent to change the caller's copy of the data
    // Destructor
    ~Wavelets();
    // Assignment (copy assignment constructor)
    Wavelets& operator=(const Wavelets &rhs) {
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
      }
      return *this;
    }

    // **********
    // Methods
    // **********
    void forward();
    void soft_threshold(float beta, int do_thresh_appcoeffs = 1);
    void circshift(int sr, int sc, int inplace = 1);
    void inverse();
    float norm2sq();
    float norm1();
    float* get_image();
    int get_image(float* img);
    float* get_coeffs();
    void print_informations();


}; // Do not forget this semicolon



/// Constructor : default
Wavelets::Wavelets(void) : d_image(NULL), Nr(0), Nc(0), nlevels(1), d_coeffs(NULL), do_cycle_spinning(0), d_tmp(NULL), current_shift_r(0), current_shift_c(0), do_swt(0)
{
}


/// Constructor :  Wavelets from image
Wavelets::Wavelets(float* img, int Nr, int Nc, const char* wname, int levels, int memisonhost, int do_cycle_spinning, int do_swt) : d_image(NULL), Nr(Nr), Nc(Nc), nlevels(levels), d_coeffs(NULL), do_cycle_spinning(do_cycle_spinning), d_tmp(NULL), current_shift_r(0), current_shift_c(0), do_swt(do_swt)
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
  int hlen = w_compute_filters(wname, 1, do_swt);
  if (hlen == 0) {
      printf("ERROR: unknown wavelet name %s\n", wname);
      exit(1); // ...
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

//~ // NEW - Wavelets from coefficients (already on device -- TODO : handle on host)
//~ Wavelets::Wavelets(float** d_thecoeffs, int Nr, int Nc, const char* wname, int levels, int do_cycle_spinning) : d_image(NULL), Nr(Nr), Nc(Nc), nlevels(levels), d_coeffs(NULL), do_cycle_spinning(do_cycle_spinning), d_tmp(NULL), current_shift_r(0), current_shift_c(0), do_swt(0)
  //~ {
//~
  //~ if (nlevels < 1) {
    //~ puts("Warning: cannot initialize wavelet coefficients with nlevels < 1. Forcing nlevels = 1");
    //~ nlevels = 1;
  //~ }
//~
  //~ float* d_theimage;
  //~ cudaMalloc(&d_theimage, Nr*Nc*sizeof(float));
  //~ cudaMemset(d_theimage, 0, Nr*Nc*sizeof(float)); // should not be necessary
  //~ this->d_image = d_theimage;
//~
  //~ this->d_coeffs = d_thecoeffs; // No copy ?
//~
  //~ float* d_tmp_new;
  //~ cudaMalloc(&d_tmp_new, Nr*Nc*sizeof(float));
  //~ this->d_tmp = d_tmp_new;
//~
  //~ float** d_appcoeffs_new;
  //~ d_appcoeffs_new = w_create_appcoeffs_buffer(Nr, Nc, nlevels, do_swt);
  //~ this->d_appcoeffs = d_appcoeffs_new;
//~
  //~ // !
  //~ int Nr2 = Nr/w_ipow2(nlevels), Nc2 = Nc/w_ipow2(nlevels);
  //~ cudaMemcpy(d_appcoeffs_new[nlevels-1], d_thecoeffs[0], Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToDevice);
//~
//~ }
//~ // ENDOF new


/// Copy
//~ Wavelets::Wavelets(const Wavelets &W) { // FIXME : I don't understand the copy order. Wouldn't be the opposite ?
  //~ Nr = W.Nr;
  //~ Nc = W.Nc;
  //~ nlevels = W.nlevels;
  //~ do_cycle_spinning = W.do_cycle_spinning;
  //~ current_shift_r = W.current_shift_r;
  //~ current_shift_c = W.current_shift_c;
  //~ float* d_new_image, *d_new_tmp;
  //~ float** d_new_coeffs, **d_new_appcoeffs;
  //~ size_t sz = Nr*Nc*sizeof(float);
  //~ cudaMalloc(&d_new_image, sz);
  //~ d_new_coeffs = w_create_coeffs_buffer(Nr, Nc, nlevels);
  //~ d_new_appcoeffs = w_create_appcoeffs_buffer(Nr, Nc, nlevels);
  //~ cudaMalloc(&d_new_tmp, sz);
  //~ cudaMemcpy(d_image, d_new_image, sz, cudaMemcpyDeviceToDevice);
  //~ w_copy_coeffs_buffer(d_coeffs, d_new_coeffs, Nr, Nc, nlevels);
  //~ w_copy_appcoeffs_buffer(d_appcoeffs, d_new_appcoeffs, Nr, Nc, nlevels);
  //~ cudaMemcpy(d_tmp, d_new_tmp, sz, cudaMemcpyDeviceToDevice);
//~ }

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
        #if defined(DO_SEPARABLE) && DO_SEPARABLE
            if (!do_swt) w_forward_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_forward_swt_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        #else
            if (!do_swt) w_forward(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_forward_swt(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        #endif
    }
}
/// Method : inverse
void Wavelets::inverse(void) {
    if ((hlen == 2) && (!do_swt)) haar_inverse2d(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels);
    else {
        #if defined(DO_SEPARABLE) && DO_SEPARABLE
            if (!do_swt) w_inverse_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_inverse_swt_separable(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        #else
            w_compute_filters(wname, -1, do_swt); // TODO : dedicated inverse coeffs to avoid this computation ?
            if (!do_swt) w_inverse(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
            else w_inverse_swt(d_image, d_coeffs, d_tmp, Nr, Nc, nlevels, hlen);
        #endif
        if (do_cycle_spinning) {
            circshift(-current_shift_r, -current_shift_c, 1);
        }
    }
}

/// Method : soft thresholding
void Wavelets::soft_threshold(float beta, int do_thresh_appcoeffs) {
  wavelets_call_soft_thresh(d_coeffs, beta, Nr, Nc, nlevels, do_swt, do_thresh_appcoeffs);
}

/// Method : circular shift
// If inplace = 1, the result is in d_image ; otherwise result is in d_tmp.
void Wavelets::circshift(int sr, int sc, int inplace) {
  wavelets_call_circshift(d_image, d_tmp, Nr, Nc, sr, sc, inplace);
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
float* Wavelets::get_image(void) {
  float* res = (float*) calloc(Nr*Nc, sizeof(float));
  cudaMemcpy(res, d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
  return res;
}

/// Method : get the image from device, on a previously allocated image
int Wavelets::get_image(float* img) {
  cudaMemcpy(img, d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
  return 0;
}

/// Method : give some informations on the wavelet
void Wavelets::print_informations() {

    const char* state[2] = {"no", "yes"};
    puts("------------- Wavelet transform infos ------------");
    printf("Wavelet name : %s\n", wname);
    printf("Number of levels : %d\n", nlevels);
    printf("Stationary WT : %s\n", state[do_swt]);
    printf("Cycle spinning : %s\n", state[do_cycle_spinning]);
    printf("Separable transform : %s\n", state[DO_SEPARABLE]);

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
    puts("--------------------------------------------------");


}





/*
// Inline operators that should not be members
 // operator += (member !) is more efficient since there is no copy...
inline Wavelets operator+(Wavelets lhs, const Wavelets &rhs) {
    lhs += rhs;
    return lhs;
}
* */



/// ****************************************************************************
/// **********************  Entry point ****************************************
/// ****************************************************************************



int main(int argc, char **argv) {

    // Read image
    int Nr = 512, Nc = 512;
    float* img = read_dat_file_float("lena.dat", Nr*Nc);

    // Configure DWT
    const char wname[128] = "haar";
    int nlevels = 4;
    int do_swt = 0;
    int do_cycle_spinning = 0;


    // ---
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart); cudaEventCreate(&tstop);
    float elapsedTime;
    // ---

    Wavelets W(img, Nr, Nc, wname, nlevels, 1, do_cycle_spinning, do_swt); CUDACHECK;
    W.print_informations();
    nlevels = W.nlevels;

    cudaEventRecord(tstart, 0);
    //---
    W.forward(); CUDACHECK;
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Forward WT (%d levels) took %lf ms\n", nlevels, elapsedTime);


    int nels;
    if (!(W.do_swt)) nels = Nr/w_ipow2(nlevels) * Nc/w_ipow2(nlevels);
    else nels = Nr*Nc;
    float* appcoeff = (float*) calloc(nels, sizeof(float));
    cudaMemcpy(appcoeff, /*W.d_coeffs[0]*/ W.d_coeffs[3*(nlevels-1)+3], nels*sizeof(float), cudaMemcpyDeviceToHost);
    write_dat_file_float("res.dat", appcoeff, nels);
    //~ return 0;

/*
    printf("Before threshold : L1 = %e\n", W.norm1());
    cudaEventRecord(tstart, 0);
    //---
    W.soft_threshold(60.0); CUDACHECK;
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Soft thresh (%d levels) took %lf ms\n", nlevels, elapsedTime);
    printf("After threshold : L1 = %e\n", W.norm1());

*/



    cudaMemset(W.d_image, 0, Nr*Nc*sizeof(float)); // To ensure that d_image is actually the result of W.inverse
    cudaEventRecord(tstart, 0);
    //---
    W.inverse(); CUDACHECK;
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Inverse WT (%d levels) took %lf ms\n", nlevels, elapsedTime);

    cudaMemcpy(img, W.d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
    write_dat_file_float("res.dat", img, Nr*Nc);

    return 0;

}




/// ----------------------------------------------------------------------------
/// -------------------------- Wrapper for shared lib --------------------------
/// ----------------------------------------------------------------------------
/*

extern "C" {
    int wrap_forward_haar(float* image, int Nr, int Nc, int nlevels, float*** coeffs, int do_cycle_spinning);
    int wrap_inverse_haar(float** image, int Nr, int Nc, int nlevels, float** coeffs, int do_cycle_spinning);
    int wrap_forward_softthresh_inverse(float* image, int Nr, int Nc, int nlevels, int do_cycle_spinning, float beta);
}

int wrap_forward_haar(float* image, int Nr, int Nc, int nlevels, float*** coeffs, int do_cycle_spinning) {
    // Build the Wavelet class
    Wavelets W(image, Nr, Nc, "haar", nlevels, 1, do_cycle_spinning);
    W.forward();
    // Retrieve the coefficients from device
    float** thecoeffs = (float**) calloc(3*nlevels+1, sizeof(float*));
    int Nr2 = Nr, Nc2 = Nc;
    for (int i = 0; i < nlevels; i++) {
        Nr2 /= 2; Nc2 /= 2;
        for (int j = 1; j <= 3; j++) {
            thecoeffs[3*i+1] = (float*) calloc(Nr2*Nc2, sizeof(float));
            cudaMemcpy(thecoeffs[3*i+j], W.d_coeffs[3*i+j], Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    // App coeff
    thecoeffs[0] = (float*) calloc(Nr2*Nc2, sizeof(float));
    cudaMemcpy(thecoeffs[0], W.d_appcoeffs[nlevels-1], Nr2*Nc2*sizeof(float), cudaMemcpyDeviceToHost);

    *coeffs = thecoeffs;
    return 0;
}

int wrap_inverse_haar(float** image, int Nr, int Nc, int nlevels, float** coeffs, int do_cycle_spinning) {
    // Transfer the coeffs into device
    int Nr2 = Nr, Nc2 = Nc;
    float** d_coeffs = (float**) calloc(3*nlevels+1, sizeof(float));
    for (int i = 0; i < nlevels; i++) {
        Nr2 /= 2; Nc2 /= 2;
        for (int j = 1; j <= 3; j++) {
            cudaMalloc(&(d_coeffs[3*i+j]), Nr2*Nc2*sizeof(float));
            cudaMemcpy(d_coeffs[3*i+j], coeffs[3*i+j], Nr2*Nc2*sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    cudaMalloc(&(d_coeffs[0]), Nr2*Nc2*sizeof(float));
    cudaMemcpy(d_coeffs[0], coeffs[0], Nr2*Nc2*sizeof(float), cudaMemcpyHostToDevice);

    // Build the Wavelet class from these coefficients
    Wavelets W(d_coeffs, Nr, Nc, "haar", nlevels, do_cycle_spinning);
    W.inverse();

    // Retrieve the image
    float* theimage = (float*) calloc(Nr*Nc, sizeof(float));
    cudaMemcpy(theimage, W.d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
    *image = theimage;

    return 0;
}

// in-place...
int wrap_forward_softthresh_inverse(float* image, int Nr, int Nc, int nlevels, int do_cycle_spinning, float beta) {
    Wavelets W(image, Nr, Nc, "haar", nlevels, 1, do_cycle_spinning);
    W.forward();
    W.soft_threshold(beta);
    W.inverse();
    W.get_image(image);
    return 0;
}

// TODO : wrap norms...

/// Test : can we wrap the whole class ?

int wrap_test(float* image, int Nr, int Nc, int nlevels, int do_cycle_spinning, void* Wstruct) {

    //~ Wavelets* foo = static_cast<Wavelets*>(Wstruct);

    //~ Wavelets W(image, Nr, Nc, nlevels, 1, do_cycle_spinning);
    //~ Wstruct =  (void*) W;
    return 0;
}

*/










