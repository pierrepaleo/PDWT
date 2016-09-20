#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import scipy.misc
from array import array


def binary_write(arr, output_filename, fmt='f'):
    output_file = open(output_filename, 'wb')
    float_array = array(fmt, arr.ravel())
    float_array.tofile(output_file)
    output_file.close()


def generateLena(Nr=512, Nc=512):
    Nr, Nc = min(Nr, 512), min(Nc, 512)
    l = scipy.misc.lena().astype(np.float32)
    l = l[:Nr, :Nc]
    binary_write(l, "lena.dat", fmt="f")


if __name__ == '__main__':

    nargs = len(sys.argv)-1
    Nr = int(sys.argv[1]) if nargs >= 1 else 512
    Nc = int(sys.argv[2]) if nargs >= 2 else 512
    generateLena(Nr, Nc)
