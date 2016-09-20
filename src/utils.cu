#include "utils.h"


int w_iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


int w_ipow2(int a) {
    return 1 << a;
}


int w_ilog2(int i) {
    int l = 0;
    while (i >>= 1) {
        ++l;
    }
    return l;
}


void w_swap_ptr(DTYPE** a, DTYPE** b) {
    DTYPE* tmp = *a;
    *a = *b;
    *b = tmp;
}
