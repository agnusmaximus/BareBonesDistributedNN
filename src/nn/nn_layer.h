#ifndef _NN_LAYER_
#define _NN_LAYER_

#include <iostream>
#include <cassert>
#include <cblas.h>
#include <vector>
#include <random>
#include "../mnist/mnist.h"
#include "../util/util.h"

class NNLayer;

class NNLayer {
 public:

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    NNLayer(int batchsize, int n_rows, int n_cols, bool is_input, bool is_output, int step) {
	std::cout << "Initializing NNLayer of dimension " << n_rows << "x" << n_cols << std::endl;
	weights = S = Z = F = NULL;
	next = prev = NULL;
	this->step = step;
	this->batchsize = batchsize;
	this->n_rows = n_rows;
	this->n_cols = n_cols;
	this->is_input = is_input;
	this->is_output = is_output;
	distribution = std::normal_distribution<double>(-1, 1);

	if (is_input) {
	    AllocateMemory(&input, n_rows*batchsize);
	}
	AllocateMemory(&S, n_rows*batchsize);

	// We add +1 for the bias column.
	AllocateMemory(&Z, (n_rows+1)*batchsize);
	AllocateMemory(&F, n_rows*batchsize);

	if (!is_output) {

	    // We add +1 for the bias weights.
	    int n_rows_to_allocate = is_input ? n_rows : n_rows+1;
	    AllocateMemory(&weights, n_rows * n_cols);
	    InitializeGaussian(weights, n_rows_to_allocate * n_cols);
	}
    }

    void WireLayers(NNLayer *prev, NNLayer *next) {
	this->next = next;
	this->prev = prev;
    }

    void ForwardPropagate(uchar **images) {
	if (is_input) {
	    // Compute S = Input * W
	    MNISTImageToInput(batchsize, images, input);
	    MatrixMultiply(input, weights, next->S,
			   batchsize, n_cols, n_rows,
			   n_rows, n_cols, n_cols);
	}
	else {
	    // Compute Z_i = f(S_i)
	    ReluActivation(S, Z, batchsize, n_rows, n_rows, n_rows+1);

	    if (is_output) {
		return;
	    }

	    // Compute S_j = Z_i W_i
	    MatrixMultiply(Z, weights, next->S,
			   batchsize, n_cols, n_rows+1,
			   n_rows+1, n_cols, n_cols);

	    // Compute F_i = f'_i(S_i)^T
	    ReluActivationGradient(S, F, batchsize, n_rows, n_rows, n_rows);
	}
	next->ForwardPropagate(images);
    }

    ~NNLayer() {
	if (weights != NULL) free(weights);
	if (S != NULL) free(S);
	if (Z != NULL) free(Z);
	if (F != NULL) free(F);
	if (input != NULL) free(input);
    }

 private:

    // Note that n_cols does account for the implicit column of noes
    // for the bias.
    int n_rows, n_cols, batchsize, step;
    bool is_input, is_output;
    double *weights, *S, *Z, *F, *input;
    NNLayer *next, *prev;

    void InitializeGaussian(double *ptr, int n_elements) {
	for (int i = 0; i < n_elements; i++) {
	    ptr[i] = distribution(generator);
	}
	for (int i = 0; i < n_rows; i++) {
	    ptr[i * n_cols + n_cols - 1] = 1;
	}
    }
};

#endif
