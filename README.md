## Plugable Parallel DWT

PPDWT is (yet another) implementation of the Discrete Wavelet Transform (DWT) on GPU.
It primarily aims to be fast, simple and versatile for an easy integration in a bigger project.

There are a few others parallel implementations of DWT, but most of them are not designed to be embedded in a project, or are specific to certain wavelets (for example (5, 3) or (9, 7)).
In this implementation, the DWT is readily computed from the provided analysis and synthesis filters.
You can either choose filters from standard families (Daubechies, symmetric, (reverse) biorthogonal, Coiffman), or design your own filters.


## Features

* 2D transform (1D and 3D coming soon)
* Separable and non-separable transforms
* 72 available separable wavelets (i.e 288 filters)
* Thresholding and norms functions
* Random shift utility for translation-invariant denoising

All the transforms are computed with the **periodic boundary extension** (the dimensions are halved at each scale).

## Current limitations

* An image can be transformed up to a scale "n" if its dimensions are a multiple of 2^n.


## Installation

### Dependencies

You need the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), and of course a NVIDIA GPU.

### Compilation

The Makefile should build smoothly the example :

```bash
make
```


## Getting started

### Running the example

To run the test, you need a raw image in 32 bits floating point precision format.
As PPDWT was primarily written for being integrated in another project, the I/O issues were not addressed : the input and output of PPDWT are float arrays.

If you have python and scipy installed, you can generate a sample file with

```bash
python generate_lena.py [Nr] [Nc]
```
where Nr, Nc are optional arguments which are the number of rows/columns of the generated image (default is 512).

You can then run an example with

```bash
./wt
```

and tune the wavelet, number of levels, etc. in the main() entry point.


### Calling PPDWT

A typical usage would be the following :

```C
// float* img = ...
int Nr = 1080; // number of rows of the image
int Nc = 1280; // number of columns of the image
int nlevels = 3;

// Compute the wavelets coefficients with the "Daubechies 7" wavelet
Wavelets W(img, Nr, Nc, "db7", nlevels, 1, 0);

// Do some thresholding on the wavelets coefficients
printf("Before threshold : L1 = %e\n", W.norm1());
W.soft_threshold(10.0);
printf("After threshold : L1 = %e\n", W.norm1());

// Inverse the DWT and retrieve the image
W.inverse();
W.get_image(img);
```


