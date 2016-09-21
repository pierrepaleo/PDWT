#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wt.h"
#include "io.h"


///
/// This file shows how to use PDWT with some examples.
///

void print_examples() {
    puts("---------------------- PDWT examples ----------------------");
    puts("1 \t Forward DWT");
    puts("2 \t Forward and inverse DWT, \"perfect reconstruction\"");
    puts("3 \t Forward DWT, threshold and inverse DWT");
    puts("0 \t Exit");
    puts("------------------------------------------------------------");
}


int request_user_choice() {
    char input[512];
    puts("What do you want to do ?");
    prompt_text(input);
    int res = strtol(input, NULL, 10);
    return res;
}

void prompt_wavelet(char* wname, int* nlevels, int* do_separable, int* do_swt) {
    char input[512];
    puts("Wavelet name [default: haar] :");
    prompt_text(wname);
    puts("Number of levels :");
    prompt_text(input);
    *nlevels = strtol(input, NULL, 10);
    puts("Separable transform ? [default: 1]");
    prompt_text(input);
    *do_separable = strtol(input, NULL, 10);
    puts("Stationary (undecimated) transform ? [default: 0]");
    prompt_text(input);
    *do_swt = strtol(input, NULL, 10);
}




int main(int argc, char **argv) {

    // Read image
    int Nr = 512, Nc = 512;
    float* img = read_dat_file_float("lena.dat", Nr*Nc);
    if (img == NULL) exit(1);
    //~ Nr = 1; // uncomment for 1D transform
    int what = 0;
    char wname[128];
    int nlevels,do_separable = 1, do_swt = 0;
    int do_cycle_spinning = 0;

    if (argc < 4) {
        printf("Usage: %s <action> <wavelet> <levels> <separable> <stationary>\n", argv[0]);
        print_examples();
        what = request_user_choice();
        if (what == 0) return 0;
        prompt_wavelet(wname, &nlevels, &do_separable, &do_swt);
    }
    else {
        what = atoi(argv[1]);
        strncpy(wname, argv[2], 128);
        nlevels = atoi(argv[3]);
        if (argc >= 5) do_separable = atoi(argv[4]);
        if (argc >= 6) do_swt = atoi(argv[5]);

    }
    if (what == 0) return 0;

    // Create the wavelet
    Wavelets W(img, Nr, Nc, wname, nlevels, 1, do_separable, do_cycle_spinning, do_swt);
    W.print_informations();
    nlevels = W.winfos.nlevels;

    // Example of custom filter (here: LeGall 9/7 lossy wavelet)
    /*
    int len = 10;
    float* filter1, *filter2, *ifilter1, *ifilter2;
    filter1 = (float*) calloc(len, sizeof(float));
    filter2 = (float*) calloc(len, sizeof(float));
    ifilter1 = (float*) calloc(len, sizeof(float));
    ifilter2 = (float*) calloc(len, sizeof(float));

    filter1[0] = 0.026748757411 ;
    filter1[1] = -0.016864118443;
    filter1[2] = -0.078223266529;
    filter1[3] = 0.266864118443 ;
    filter1[4] = 0.602949018236 ;
    filter1[5] = 0.266864118443 ;
    filter1[6] = -0.078223266529;
    filter1[7] = -0.016864118443;
    filter1[8] = 0.026748757411 ;

    filter2[0] = 0              ;
    filter2[1] = 0.091271763114 ;
    filter2[2] = -0.057543526229;
    filter2[3] = -0.591271763114;
    filter2[4] =    1.11508705  ;
    filter2[5] = -0.591271763114;
    filter2[6] = -0.057543526229;
    filter2[7] = 0.091271763114 ;
    filter2[8] = 0              ;

    ifilter1[0] = 0               ;
    ifilter1[1] = -0.091271763114 ;
    ifilter1[2] = -0.057543526229 ;
    ifilter1[3] = 0.591271763114  ;
    ifilter1[4] = 1.11508705      ;
    ifilter1[5] = 0.591271763114  ;
    ifilter1[6] = -0.057543526229 ;
    ifilter1[7] = -0.091271763114 ;
    ifilter1[8] = 0               ;

    ifilter2[0] = 0.026748757411 ;
    ifilter2[1] = 0.016864118443 ;
    ifilter2[2] = -0.078223266529;
    ifilter2[3] = -0.266864118443;
    ifilter2[4] = 0.602949018236 ;
    ifilter2[5] = -0.266864118443;
    ifilter2[6] = -0.078223266529;
    ifilter2[7] = 0.016864118443 ;
    ifilter2[8] = 0.026748757411 ;

    W.set_filters_forward(len, filter1, filter2);
    W.set_filters_inverse(ifilter1, ifilter2);
    */


    // Perform forward WT with current configuration
    W.forward();
    puts("Forward OK");

    float* thecoeffs = (float*) calloc(Nr*Nc, sizeof(float)); // larger than needed
    int nels = W.get_coeff(thecoeffs, 0); //3*(nlevels-1)+3);
    write_dat_file_float("res.dat", thecoeffs, nels);
    if (what == 1) {
        printf("Approximation coefficients (level %d) are stored in res.dat\n", nlevels);
        return 0;
    }

    if (what == 3) {
        printf("Before threshold : L1 = %e\n", W.norm1());
        W.soft_threshold(90.0, 0, 0);
        printf("After threshold : L1 = %e\n", W.norm1());
    }


    // Perform inverse WT with current configuration.
    // Ensures that the result is actually the inverse
    float* dummy = (float*) calloc(Nr*Nc, sizeof(float));
    W.set_image(dummy, 0);

    W.inverse();
    puts("Inverse OK");

    W.get_image(img);
    write_dat_file_float("res.dat", img, Nr*Nc);
    puts("Wrote result in res.dat");

    return 0;

}
