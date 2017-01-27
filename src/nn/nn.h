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
	layers.push_back(new NNLayer(batchsize, batchsize, IMAGE_X*IMAGE_Y, true,
				     false, 0));
	for (int i = 0; i < params->GetLayers().size()-1; i++) {
	    std::pair<int, int> layer = params->GetLayers()[i];
	    std::pair<int, int> next_layer = params->GetLayers()[i+1];
	    layers.push_back(new NNLayer(batchsize,
					 layer.second, next_layer.second, false,
					 false, 0));
	}
	layers.push_back(new NNLayer(batchsize, N_CLASSES, -1, false,
				     true, 0));
	for (int i = 0; i < layers.size(); i++) {
	    NNLayer *prev = i == 0 ? NULL : layers[i-1];
	    NNLayer *next = i == layers.size()-1 ? NULL : layers[i+1];
	    layers[i]->WireLayers(prev, next);
	}
    }

    void ForwardPropagate(uchar **images) {
	layers[0]->ForwardPropagate(images);
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
    int batch_size = 128;
    params->AddLayer(batch_size, IMAGE_X*IMAGE_Y);
    params->AddLayer(IMAGE_X*IMAGE_Y, 400);
    params->AddLayer(400, N_CLASSES);
    NN *nn = new NN(params, batch_size);
    int number_of_images, image_size;
    uchar **images = read_mnist_images(TRAINING_IMAGES, number_of_images, image_size);
    nn->ForwardPropagate(images);

    delete nn;
    delete params;

    std::cout << "Done testing nn!" << std::endl;
}

#endif
