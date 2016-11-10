#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "filters.h"


struct w_info {
    // Number of dimensions. For now only 2D and (batched) 1D are supported
    int ndims;
    // Number of rows and columns of the image (for 1D : Nr = 1)
    int Nr;
    int Nc;
    // Wavelet transform related informations
    int nlevels;            // Number of decomposition levels
    int do_swt;             // Do Stationary (Undecimated) Wavelet Transform
    int hlen;               // "Filter" length
};

int w_iDivUp(int a, int b);

int w_ipow2(int a);

int w_ilog2(int i);

void w_div2(int* N);

void w_swap_ptr(DTYPE** a, DTYPE** b);

#endif
