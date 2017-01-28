#include <iostream>
#include <cblas.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define BUMP 1e-10
#define INF std::numeric_limits<double>::infinity()

void AllocateMemory(double **ptr, int sz) {
    *ptr = (double *)malloc(sizeof(double) * sz);
    if (!*ptr) {
	std::cout << "Error allocating memory." << std::endl;
	exit(-1);
    }
    memset(*ptr, 0, sizeof(double) * sz);
}

// C = A*alpha + b*beta
void MatrixAdd(double *A, double *B, double *C, double alpha, double beta,
	       int n_rows, int n_cols, int lda, int ldb, int ldc) {
    for (int row = 0; row < n_rows; row++) {
	for (int col = 0; col < n_cols; col++) {
	    C[row*ldc+col] = A[row*lda+col] * alpha + B[row*ldb+col] * beta;
	}
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
    double s = 0, maximum = -INF;
    for (int i = 0; i < length; i++) {
	maximum = std::max(maximum, in[i]);
    }
    for (int i = 0; i < length; i++) {
	s += exp(in[i]-maximum);
    }
    for (int i = 0; i < length; i++) {
	out[i] = exp(in[i]-maximum) / s;
    }
}

double LogDot(double *a, double *b, int length) {
    double r = 0;
    for (int i = 0; i < length; i++) {
	r += -log(a[i]+BUMP) * b[i];
    }
    return r;
}

double Argmax(double *a, int length) {
    int index = -INF;
    double maxi =-INF;
    for (int i = 0; i < length; i++) {
	if (a[i] > maxi) {
	    index = i;
	    maxi = a[i];
	}
    }
    return index;
}

void PrintMatrix(double *data, int h, int w) {
    for (int i = 0; i < h; i++) {
	for (int j = 0; j < w; j++) {
	    std::cout << data[i*w + j] << " ";
	}
	std::cout << std::endl;
    }
}
