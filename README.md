## Parallel DWT

PDWT is a parallel implementation of the Discrete Wavelet Transform (DWT).
This implementation in CUDA targets Nvidia GPUs.

PDWT primarily aims at being fast, simple and versatile for an easy integration in a bigger project.
For example, the easy interface and thresholding functions make it interesting for sparse regularization of inverse problems.


## Features

* 1D and 2D transform, multi-levels, arbitrary sizes
* Support of batched 1D transform
* Separable and non-separable transforms
* DWT and SWT, both in separable/nonseparable mode
* 72 available separable wavelets
* Custom wavelets can be defined
* Thresholding and norms utilities
* Random shift utility for translation-invariant denoising
* Simple interface (see examples)
* Python binding available
* Results compatible with Matlab wavelet toolbox / Python pywt.

All the transforms are computed with the **periodic boundary extension** (the dimensions are halved at each scale).

## Current limitations

* 3D is not handled at the moment.
* Only the periodic boundary extension is implemented.
* The parallel part is implemented in CUDA, so only Nvidia GPUs can be used.


## Installation

### Dependencies

You need the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), and of course a NVIDIA GPU.

### Compilation

The Makefile should build smoothly the example :

```bash
make demo
```


## Getting started

### Running the example

To run the test, you need a raw image in 32 bits floating point precision format.
As PDWT was primarily written for data crunching, the I/O part is not addressed : the input and output of PDWT are float arrays.

If you have python and scipy installed, you can generate an image input file with

```bash
python generate_lena.py [Nr] [Nc]
```
where Nr, Nc are optional arguments which are the number of rows/columns of the generated image (default is 512).

You can then run an example with

```bash
make demo
./build/demo
```

and tune the wavelet, number of levels, etc. in the prompt.


### Calling PDWT

A typical usage would be the following :

```C
#include "wt.h"

// ...
// float* img = ...
int Nr = 1080; // number of rows of the image
int Nc = 1280; // number of columns of the image
int nlevels = 3;

// Compute the wavelets coefficients with the "Daubechies 7" wavelet
Wavelets W(img, Nr, Nc, "db7", nlevels);
W.forward();

// Do some thresholding on the wavelets coefficients
float norm1 = W1.norm1();
printf("Before threshold : L1 = %e\n", norm1);
W.soft_threshold(10.0);
norm1 = W1.norm1();
printf("After threshold : L1 = %e\n", norm1);

// Inverse the DWT and retrieve the image from the GPU
W.inverse();
W.get_image(img);
```


