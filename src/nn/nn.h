#ifndef _NN_
#define _NN_

#include <iostream>
#include <vector>
#include "nn_params.h"
#include "nn_layer.h"
#include "../mnist/mnist.h"

class NN {
 public:
    NN(NNParams *params, int batchsize) {
	params->Validate(batchsize, N_CLASSES);
    }

    ~NN() {
	for (int i = 0; i < layers.size(); i++) {
	    delete layers[i];
	}
    }

 private:
    std::vector<NNLayer *> layers;
};

void test_nn() {

    std::cout << "Testing nn..." << std::endl;

    NNParams *params = new NNParams();
    params->AddLayer(128, 300);
    params->AddLayer(300, 400);
    params->AddLayer(400, 10);

    NN *nn = new NN(params, 128);

    delete nn;
    delete params;

    std::cout << "Done testing nn!" << std::endl;
}

#endif
