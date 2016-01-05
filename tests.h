#ifndef TESTS_H
#define TESTS_H

char* wt_file_name(const char* wname, int nlevels, const char* suffix);

int test_forward(float* img, int Nr, int Nc, const char* wname, int nlevels, int verbose = 1);

#endif
