#ifndef WT_H
#define WT_H


void wavelets_call_soft_thresh(float** d_coeffs, float beta, int Nr, int Nc, int nlevels, int do_swt, int do_thresh_appcoeffs);

void wavelets_call_circshift(float* d_image, float* d_image2, int Nr, int Nc, int sr, int sc, int inplace = 1);

float** w_create_coeffs_buffer(int Nr, int Nc, int nlevels, int do_swt);

void w_free_coeffs_buffer(float** coeffs, int nlevels);

void w_copy_coeffs_buffer(float** dst, float** src, int Nr, int Nc, int nlevels, int do_swt);


class Wavelets {
  public:
    // Members
    // --------
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
    int do_separable; // 1 if performing separable WT

    // Operations
    // -----------
    // Default constructor
    Wavelets();
    // Constructor : Wavelets from image
    Wavelets(float* img, int Nr, int Nc, const char* wname, int levels, int memisonhost=1, int do_separable = 1, int do_cycle_spinning = 0, int do_swt = 0);
    // Constructor : Wavelets from coeffs
    //~ Wavelets(float** d_thecoeffs, int Nr, int Nc, const char* wname, int levels, int do_cycle_spinning);
    // Class copy (constructor)
    Wavelets(const Wavelets &W);  // Pass by non-const reference ONLY if the function will modify the parameter and it is the intent to change the caller's copy of the data
    // Destructor
    ~Wavelets();
    // Assignment (copy assignment constructor)
    Wavelets& operator=(const Wavelets &rhs);

    // Methods
    // -------
    void forward();
    void soft_threshold(float beta, int do_thresh_appcoeffs = 1);
    void circshift(int sr, int sc, int inplace = 1);
    void inverse();
    float norm2sq();
    float norm1();
    int get_image(float* img);
    void print_informations();
    int get_coeff(float* coeff, int num);

};




#endif

