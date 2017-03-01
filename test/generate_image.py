#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import scipy.misc
from array import array
try:
    img = scipy.misc.lena().astype(np.float32)
except: # new versions of scipy
    img = scipy.misc.ascent().astype(np.float32)


def binary_write(arr, output_filename, fmt='f'):
    output_file = open(output_filename, 'wb')
    float_array = array(fmt, arr.ravel())
    float_array.tofile(output_file)
    output_file.close()


def generateImage(Nr=512, Nc=512):
    Nr, Nc = min(Nr, 512), min(Nc, 512)
    l = img[:Nr, :Nc]
    binary_write(l, "image.dat", fmt="f")


if __name__ == '__main__':

    nargs = len(sys.argv)-1
    Nr = int(sys.argv[1]) if nargs >= 1 else 512
    Nc = int(sys.argv[2]) if nargs >= 2 else 512
    generateImage(Nr, Nc)
