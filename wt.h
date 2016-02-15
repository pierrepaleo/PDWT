#ifndef WT_H
#define WT_H


// Possible states of the Wavelet class.
// It prevents, for example, W.inverse() from being run twice (since W.d_coeffs[0] is modified)
typedef enum w_state {
    W_INIT,      // The class has just been initialized (coeffs not computed)
    W_FORWARD,   // W.forward() has just been performed (coeffs computed)
    W_INVERSE,   // W.inverse() has just been performed (d_image modified, coeffs modified !)
    W_THRESHOLD  // The coefficients have been modified
} w_state;


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
    w_state state;

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
    void hard_threshold(float beta, int do_thresh_appcoeffs = 1);
    void shrink(float beta, int do_thresh_appcoeffs = 1);
    void circshift(int sr, int sc, int inplace = 1);
    void inverse();
    float norm2sq();
    float norm1();
    int get_image(float* img);
    void print_informations();
    int get_coeff(float* coeff, int num);
    void set_image(float* img, int mem_is_on_device = 0);

};




#endif

