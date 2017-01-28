#include <iostream>
#include <cblas.h>
#include <math.h>
#include <vector>
#include <algorithm>

void AllocateMemory(double **ptr, int sz) {
    *ptr = (double *)malloc(sizeof(double) * sz);
    if (!*ptr) {
	std::cout << "Error allocating memory." << std::endl;
	exit(-1);
    }
}

// C = A*B
// A = mxk, B = kxn, c = mxn
// mm - leading dimension of A
// nn - leading dimension of B
// cc - leading dimension of C
void MatrixMultiply(double *A, double *B, double *C,
		    int m, int n, int k,
		    int mm, int nn, int kk) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k,
		1,
		A, mm,
		B, nn,
		1,
		C, kk);
}

void ReluActivation(double *in, double *out,
		    int n_rows, int n_cols,
		    int ld_in, int ld_out) {
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols; j++) {
	    out[i*ld_out+j] = std::max((double)0, in[i*ld_in+j]);
	}
    }
}

void ReluActivationGradient(double *in, double *out,
			    int n_rows, int n_cols,
			    int ld_in, int ld_out) {
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols; j++) {
	    out[i*ld_out+j] = in[i*ld_in+j] < 0 ? 0 : 1;
	}
    }
}

void Softmax(double *in, double *out, int length) {
    double s = 0, maximum = -1000000000;
    for (int i = 0; i < length; i++) {
	maximum = std::max(maximum, in[i]);
    }
    for (int i = 0; i < length; i++) {
	s += exp(in[i]-maximum);
    }
    for (int i = 0; i < length; i++) {
	out[i] = in[i] / s;
    }
}
