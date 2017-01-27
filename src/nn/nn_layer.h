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

    NNLayer(int batchsize, int n_rows, int n_cols, bool is_input, bool is_output) {
	n_cols++;
	weights = S = Z = F = NULL;
	next = prev = NULL;
	this->batchsize = batchsize;
	this->n_cols = n_cols;
	this->n_rows = n_rows;
	this->is_input = is_input;
	this->is_output = is_output;
	distribution = std::normal_distribution<double>(-1, 1);

	if (is_input) {
	    AllocateMemory(&input, IMAGE_X*IMAGE_Y*batchsize);
	}

	AllocateMemory(&S, n_rows * n_cols);
	InitializeGaussian(S, n_rows * n_cols);
	AllocateMemory(&weights, n_rows * n_cols);
	InitializeGaussian(weights, n_rows * n_cols);
	AllocateMemory(&Z, n_rows * n_cols);
	InitializeGaussian(Z, n_rows * n_cols);
	AllocateMemory(&F, n_rows * n_cols);
	InitializeGaussian(F, n_rows * n_cols);
    }

    void WireLayers(NNLayer *prev, NNLayer *next) {
	this->next = next;
	this->prev = prev;
    }

    void ForwardPropagate(uchar **images) {
	if (is_input) {
	    InputCopyStridedToInput(images);
	}
	else if (is_output) {

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
    int n_cols, n_rows, batchsize;
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
	for (int i = 0; i < n_rows; i++) {
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
	for (int i = 0; i < n_rows; i++) {
	    assert(ptr[i * n_cols + n_cols - 1] == 1);
	}
    }

};

#endif
