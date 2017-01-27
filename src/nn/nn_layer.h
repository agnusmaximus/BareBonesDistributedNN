#ifndef _NN_LAYER_
#define _NN_LAYER_

#include <iostream>
#include <cassert>
#include <cblas.h>
#include <vector>
#include <random>
#include "../mnist/mnist.h"

class NNLayer;

class NNLayer {
 public:

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    NNLayer(int batchsize, int n_rows, int n_cols, bool is_input, bool is_output, int step) {
	n_cols++;
	weights = S = Z = F = NULL;
	next = prev = NULL;
	this->step = step;
	this->batchsize = batchsize;
	this->n_cols = n_cols;
	this->is_input = is_input;
	this->is_output = is_output;
	distribution = std::normal_distribution<double>(-1, 1);

	if (is_output) return;

	AllocateMemory(&input, IMAGE_X*IMAGE_Y*batchsize);
	AllocateMemory(&S, batchsize * n_cols);
	InitializeGaussian(S, batchsize * n_cols);
	AllocateMemory(&weights, batchsize * n_cols);
	InitializeGaussian(weights, batchsize * n_cols);
	AllocateMemory(&Z, batchsize * n_cols);
	InitializeGaussian(Z, batchsize * n_cols);
	AllocateMemory(&F, batchsize * n_cols);
	InitializeGaussian(F, batchsize * n_cols);
    }

    void WireLayers(NNLayer *prev, NNLayer *next) {
	this->next = next;
	this->prev = prev;
    }

    void ForwardPropagate(uchar **images) {
	if (is_input) {
	    // Compute S = Input * W
	    InputCopyStridedToInput(images);
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			batchsize, n_cols, IMAGE_X*IMAGE_Y,
			1,
			input, IMAGE_X*IMAGE_Y,
			weights, n_cols,
			1,
			next->S, n_cols);
	}
	else {

	}

	if (next) {
	    next->ForwardPropagate(images);
	}
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
    int n_cols, batchsize, step;
    bool is_input, is_output;
    double *weights, *S, *Z, *F, *input;
    NNLayer *next, *prev;

    void AllocateMemory(double **ptr, int sz) {
	*ptr = (double *)malloc(sizeof(double) * sz);
	if (!*ptr) {
	    std::cout << "Error allocating memory." << std::endl;
	    exit(-1);
	}
    }

    void InitializeGaussian(double *ptr, int n_elements) {
	for (int i = 0; i < n_elements; i++) {
	    ptr[i] = distribution(generator);
	}
	for (int i = 0; i < batchsize; i++) {
	    ptr[i * n_cols + n_cols - 1] = 1;
	}
    }

    void InputCopyStridedToInput(uchar **images) {
	for (int i = 0; i < batchsize; i++) {
	    for (int j = 0; j < IMAGE_X*IMAGE_Y; j++) {
		input[i * IMAGE_X*IMAGE_Y + j] = images[i][j];
	    }
	}
    }

    void AssertBiasColumnIsOne(double *ptr) {
	for (int i = 0; i < batchsize; i++) {
	    assert(ptr[i * n_cols + n_cols - 1] == 1);
	}
    }

};

#endif
