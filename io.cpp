/// ****************************************************************************
/// ****************** I/O (simple raw read/write)  ****************************
/// ****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"

float* read_dat_file_float(const char* fname, int len) {
    FILE* fid = fopen(fname, "rb");
    if (fid == NULL) {
        printf("ERROR in read_dat_file_float(): could not read %s\n", fname);
        return NULL;
    }
    float* out = (float*) calloc(len, sizeof(float));
    fread(out, len, sizeof(float), fid);
    fclose(fid);
    return out;
}

void write_dat_file_float(const char* fname, float* arr, int len) {
    FILE* fid = fopen(fname, "wb");
    if (fid == NULL) return;
    fwrite(arr, len, sizeof(float), fid);
    fclose(fid);
}

#define MAX_PROMT_LENGTH 512
int prompt_text(char* input) {
    if (!fgets(input, MAX_PROMT_LENGTH, stdin)) return -1; // reading failed
    size_t l = strlen(input);    // <string.h> must be included
    if (input[l-1] == '\n') {
        if (l > 1) { // remove trailing newline
            input[l-1] = 0;
        }
        else return 0; // blank line
    }
    else return -2; // Too long input
    return l;
}
