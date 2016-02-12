///
/// This file is intended to compare the results of PPDWT to the implementation of "pywt" (http://www.pybytes.com/pywavelets)
///


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
//~ #include <unistd.h>
#include "wt.h"
#include "io.h"




#include "tests.h"

// Do not forget the "/" at the end
#define W_SAVE_DIR "/home/paleo/Projets/public/ppdwt/results_ppdwt/"


char* wt_file_name(const char* wname, int nlevels, const char* suffix) {
    char* res = (char*) calloc(384, sizeof(char));
    char slevels[5];
    snprintf(slevels, 5, "_%d_", nlevels);
    res[0] = '\0';
    strncat(res, W_SAVE_DIR, 128);
    strncat(res, wname, 128);
    strncat(res, slevels, 5);
    strncat(res, suffix, 32);
    strncat(res, ".dat", 8);
    return res;
}



// Test the forward DWT (separable, decimated).
int test_forward(float* img, int Nr, int Nc, const char* wname, int nlevels, int verbose) {


    Wavelets W(img, Nr, Nc, wname, nlevels, 1, 1, 0, 0);
    if (verbose >= 3) W.print_informations();
    nlevels = W.nlevels;

    W.forward();

    int Nr2 = (Nr >> nlevels), Nc2 = (Nc >> nlevels);
    int nels = Nr2*Nc2;
    float* thecoeffs = (float*) calloc(nels, sizeof(float));

    char* wfile_app = wt_file_name(wname, nlevels, "app");
    char* wfile_det = wt_file_name(wname, nlevels, "det3");

    // Save the appcoeff and the last D coeff
    W.get_coeff(thecoeffs, 0);
    write_dat_file_float(wfile_app, thecoeffs, nels);
    W.get_coeff(thecoeffs, 3*(nlevels-1)+3);
    write_dat_file_float(wfile_det, thecoeffs, nels);


    free(thecoeffs);
    free(wfile_app);
    free(wfile_det);

}


int create_folder_if_not_existing(const char* fname) {
    struct stat st = {0};
    int res = -2; // already exists
    if (stat(fname, &st) == -1) res = mkdir(fname, 0755);
    return res;
}

// paths are assumed to be unix-like
int load_python_transform(char* wname, int levels, char* savedir, int do_dwt) {

    what = ??
    s = app, detN
    // ex : path/db10_3_dwt_det1.dat
    snprintf(fullname, 511, "%s/%s_%d_%s", savedir, wname, levels, whichtransform, whichcoeff)
    strncat(fullname, savedir, "/"



}










int main() {

    int verbose = 1;

    // Edit the following variables to test more wavelets
    int n_wavelets = 5;
    const char* wavelets_to_test[n_wavelets] = {"haar", "db2", "db3", "sym2", "sym3"};
    int levels_to_test[n_wavelets] = {8, 4, 3, 5, 2};


    //
    // --------
    int Nr = 512, Nc = 512;
    float* img = read_dat_file_float("lena.dat", Nr*Nc);



    for (int i = 0; i < n_wavelets; i++) {

        if (verbose >= 1) printf("Testing %s (level %d)\n", wavelets_to_test[i], levels_to_test[i]);
        test_forward(img, Nr, Nc, wavelets_to_test[i], levels_to_test[i], verbose);


    }



    free(img);
}



///
/// Verbosity levels :
///     -1 : no message at all, return only the results
///     0 : throw message when a fail occurs
///     1 : print what test is being performed
///     2 : more information
///     3 : all available information
///
