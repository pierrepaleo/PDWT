#ifndef WT_H
#define WT_H

#include "utils.h"

// Possible states of the Wavelet class.
// It prevents, for example, W.inverse() from being run twice (since W.d_coeffs[0] is modified)
typedef enum w_state {
    W_INIT,             // The class has just been initialized (coeffs not computed)
    W_FORWARD,          // W.forward() has just been performed (coeffs computed)
    W_INVERSE,          // W.inverse() has just been performed (d_image modified, coeffs modified !)
    W_THRESHOLD,        // The coefficients have been modified
    W_CREATION_ERROR,   // Error when creating the Wavelets instance
    W_FORWARD_ERROR,    // Error when computing the forward transform
    W_INVERSE_ERROR,    // Error when computing the inverse transform
    W_THRESHOLD_ERROR   // Error when thresholding the coefficients
} w_state;


class Wavelets {
  public:
    // Members
    // --------
    DTYPE* d_image;         // Image (input or result of reconstruction), on device
    DTYPE** d_coeffs;       // Wavelet coefficients, on device
    DTYPE* d_tmp;           // Temporary device array (to avoid multiple malloc/free)

    int current_shift_r;
    int current_shift_c;
    char wname[128];        // Wavelet name
    int do_separable;       // 1 if performing separable WT
    int do_cycle_spinning;  // Do image shifting for approximate TI denoising
    w_info winfos;
    w_state state;


    // Operations
    // -----------
    // Default constructor
    Wavelets();
    // Constructor : Wavelets from image
    Wavelets(DTYPE* img, int Nr, int Nc, const char* wname, int levels, int memisonhost=1, int do_separable=1, int do_cycle_spinning=0, int do_swt=0, int ndim=2);
    // Constructor: copy
    Wavelets(const Wavelets &W);// Pass by non-const reference ONLY if the function will modify the parameter and it is the intent to change the caller's copy of the data
    // Constructor : Wavelets from coeffs
    //~ Wavelets(DTYPE** d_thecoeffs, int Nr, int Nc, const char* wname, int levels, int do_cycle_spinning);
    // Destructor
    ~Wavelets();
    // Assignment (copy assignment constructor)
    // do not use !
    // Wavelets& operator=(const Wavelets &rhs);

    // Methods
    // -------
    void forward();
    void soft_threshold(DTYPE beta, int do_thresh_appcoeffs = 0, int normalize = 0);
    void hard_threshold(DTYPE beta, int do_thresh_appcoeffs = 0, int normalize = 0);
    void group_soft_threshold(DTYPE beta, int do_thresh_appcoeffs = 0, int normalize = 0);
    void shrink(DTYPE beta, int do_thresh_appcoeffs = 1);
    void proj_linf(DTYPE beta, int do_thresh_appcoeffs = 1);
    void circshift(int sr, int sc, int inplace = 1);
    void inverse();
    DTYPE norm2sq();
    DTYPE norm1();
    int get_image(DTYPE* img);
    void print_informations();
    int get_coeff(DTYPE* coeff, int num);
    void set_image(DTYPE* img, int mem_is_on_device = 0);
    void set_coeff(DTYPE* coeff, int num, int mem_is_on_device = 0);
    int set_filters_forward(char* filtername, uint len, DTYPE* filter1, DTYPE* filter2, DTYPE* filter3 = NULL, DTYPE* filter4 = NULL);
    int set_filters_inverse(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3 = NULL, DTYPE* filter4 = NULL);

    int add_wavelet(Wavelets W, DTYPE alpha=1.0f);
    __intptr_t image_int_ptr(void);
    __intptr_t coeff_int_ptr(int num);
};


#endif

