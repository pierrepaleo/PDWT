#include <stdio.h>
#include <stdlib.h>
#include "wt.h"
#include "io.h"

///
/// In this example, we profile the time to perform the forward WT, inverse, thresholding...
/// You can modify the properties of the transform (wavelet, levels, separability, SWT)
///

int main(int argc, char **argv) {

    // Read image
    int Nr = 512, Nc = 512;
    float* img = read_dat_file_float("lena.dat", Nr*Nc);

    // Configure DWT
    const char wname[128] = "db3"; //"haar";
    int nlevels = 2; //4;
    int do_separable = 1;
    int do_swt = 0;
    int do_cycle_spinning = 0;


    // ---
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart); cudaEventCreate(&tstop);
    float elapsedTime;
    // ---

    Wavelets W(img, Nr, Nc, wname, nlevels, 1, do_separable, do_cycle_spinning, do_swt);
    W.print_informations();
    nlevels = W.nlevels;

    cudaEventRecord(tstart, 0);
    //---
    W.forward();
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Forward WT (%d levels) took %lf ms\n", nlevels, elapsedTime);


    int nels;
    if (!(W.do_swt)) nels = (Nr >> nlevels) * (Nc >> nlevels);
    else nels = Nr*Nc;
    float* thecoeffs = (float*) calloc(nels, sizeof(float));
    W.get_coeff(thecoeffs, 3*(nlevels-1)+3);
    write_dat_file_float("res.dat", thecoeffs, nels);
    //~ return 0;



    cudaEventRecord(tstart, 0);
    //---
    W.soft_threshold(10.0);
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Soft thresh (%d levels) took %lf ms\n", nlevels, elapsedTime);


    cudaMemset(W.d_image, 0, Nr*Nc*sizeof(float)); // To ensure that d_image is actually the result of W.inverse
    cudaEventRecord(tstart, 0);
    //---
    W.inverse();
    //---
    cudaEventRecord(tstop, 0); cudaEventSynchronize(tstop); cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    printf("Inverse WT (%d levels) took %lf ms\n", nlevels, elapsedTime);

    cudaMemcpy(img, W.d_image, Nr*Nc*sizeof(float), cudaMemcpyDeviceToHost);
    write_dat_file_float("res.dat", img, Nr*Nc);

    return 0;

}
