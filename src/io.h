#ifndef IO_H
#define IO_H

float* read_dat_file_float(const char* fname, int len);

void write_dat_file_float(const char* fname, float* arr, int len);

int prompt_text(char* input);

#endif
