/// ****************************************************************************
/// ****************** I/O (simple raw read/write)  ****************************
/// ****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"

DTYPE* read_dat_file_DTYPE(const char* fname, int len) {
    FILE* fid = fopen(fname, "rb");
    if (fid == NULL) {
        printf("ERROR in read_dat_file_DTYPE(): could not read %s\n", fname);
        return NULL;
    }
    DTYPE* out = (DTYPE*) calloc(len, sizeof(DTYPE));
    fread(out, len, sizeof(DTYPE), fid);
    fclose(fid);
    return out;
}

void write_dat_file_DTYPE(const char* fname, DTYPE* arr, int len) {
    FILE* fid = fopen(fname, "wb");
    if (fid == NULL) return;
    fwrite(arr, len, sizeof(DTYPE), fid);
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
