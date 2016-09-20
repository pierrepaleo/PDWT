#ifndef IO_H
#define IO_H

DTYPE* read_dat_file_DTYPE(const char* fname, int len);

void write_dat_file_DTYPE(const char* fname, DTYPE* arr, int len);

int prompt_text(char* input);

#endif
