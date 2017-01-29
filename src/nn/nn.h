#ifndef _NN_
#define _NN_

#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include "nn_params.h"
#include "nn_layer.h"
#include "../mnist/mnist.h"

class NN {
 public:
    NN(NNParams *params) {
	this->batchsize = params->GetBatchsize();
	this->learning_rate = params->GetLearningRate();
	params->Validate(batchsize, N_CLASSES);
	for (int i = 0; i < params->GetLayers().size()-1; i++) {
	    std::pair<int, int> layer = params->GetLayers()[i];
	    std::pair<int, int> next_layer = params->GetLayers()[i+1];
	    layers.push_back(new NNLayer(batchsize,
					 layer.second, next_layer.second,
					 i == 0,
					 false, 0, learning_rate));
	}
	layers.push_back(new NNLayer(batchsize,
				     params->GetLayers()[params->GetLayers().size()-1].second, -1,
				     false,
				     true, 0, learning_rate));
	for (int i = 0; i < layers.size(); i++) {
	    NNLayer *prev = i == 0 ? NULL : layers[i-1];
	    NNLayer *next = i == layers.size()-1 ? NULL : layers[i+1];
	    layers[i]->WireLayers(prev, next);
	}
    }

    void Train(uchar **data, uchar *labels, int n_examples) {
	MNISTShuffleDataAndLabels(data, labels, n_examples);
	int n_features = layers[0]->Dimension();
	int n_outputs = layers[layers.size()-1]->Dimension();
	double *batch_data_placeholder = (double *)malloc(sizeof(double) * batchsize * n_features);
	double *batch_labels_placeholder = (double *)malloc(sizeof(double) * batchsize * n_outputs);
	if (!batch_data_placeholder || !batch_labels_placeholder) {
	    std::cout << "Error allocating memory for Train" << std::endl;
	    exit(-1);
	}
	for (int index = 0; index < n_examples; index += batchsize) {
	    int n_to_copy = std::min(batchsize, n_examples-index);
	    MNISTImageToInput(n_to_copy, &data[index], batch_data_placeholder);
	    MNISTOneHotLabelsToInput(n_to_copy, &labels[index], batch_labels_placeholder);
	    if (n_to_copy < batchsize) {
		memset(&batch_data_placeholder[batchsize-n_to_copy], 0, sizeof(double) * n_features * (batchsize-n_to_copy));
		memset(&batch_labels_placeholder[batchsize-n_to_copy], 0, sizeof(double) * (batchsize-n_to_copy));
	    }

	    ForwardPropagate(batch_data_placeholder);
	    BackPropagate(batch_labels_placeholder);
	}

	for (int l = 0; l < layers.size(); l++) {
	    layers[l]->IncStep();
	}

	free(batch_data_placeholder);
	free(batch_labels_placeholder);
    }

    double ComputeLoss(uchar **data, uchar *labels, int n_examples) {
	int n_features = layers[0]->Dimension();
	int n_outputs = layers[layers.size()-1]->Dimension();
	double *batch_data_placeholder = (double *)malloc(sizeof(double) * batchsize * n_features);
	double *batch_labels_placeholder = (double *)malloc(sizeof(double) * batchsize * n_outputs);
	if (!batch_data_placeholder || !batch_labels_placeholder) {
	    std::cout << "Error allocating memory for ComputeLoss" << std::endl;
	    exit(-1);
	}
	double loss = 0;
	for (int index = 0; index < n_examples; index += batchsize) {
	    int n_to_copy = std::min(batchsize, n_examples-index);
	    MNISTImageToInput(n_to_copy, &data[index], batch_data_placeholder);
	    MNISTOneHotLabelsToInput(n_to_copy, &labels[index], batch_labels_placeholder);
	    if (n_to_copy < batchsize) {
		memset(&batch_data_placeholder[batchsize-n_to_copy], 0, sizeof(double) * n_features * (batchsize-n_to_copy));
		memset(&batch_labels_placeholder[batchsize-n_to_copy], 0, sizeof(double) * (batchsize-n_to_copy));
	    }
	    loss += ComputeBatchLoss(batch_data_placeholder,
				     batch_labels_placeholder,
				     n_to_copy);
	}
	free(batch_data_placeholder);
	free(batch_labels_placeholder);
	return loss;
    }

    double ComputeErrorRate(uchar **data, uchar *labels, int n_examples) {
	int n_features = layers[0]->Dimension();
	int n_outputs = layers[layers.size()-1]->Dimension();
	double *batch_data_placeholder = (double *)malloc(sizeof(double) * batchsize * n_features);
	double *batch_labels_placeholder = (double *)malloc(sizeof(double) * batchsize * n_outputs);
	if (!batch_data_placeholder || !batch_labels_placeholder) {
	    std::cout << "Error allocating memory for ComputeErrorRate" << std::endl;
	    exit(-1);
	}
	double n_wrong = 0, n_seen = 0;
	for (int index = 0; index < n_examples; index += batchsize) {
	    int n_to_copy = std::min(batchsize, n_examples-index);
	    MNISTImageToInput(n_to_copy, &data[index], batch_data_placeholder);
	    MNISTOneHotLabelsToInput(n_to_copy, &labels[index], batch_labels_placeholder);
	    if (n_to_copy < batchsize) {
		memset(&batch_data_placeholder[batchsize-n_to_copy], 0, sizeof(double) * n_features * (batchsize-n_to_copy));
		memset(&batch_labels_placeholder[batchsize-n_to_copy], 0, sizeof(double) * (batchsize-n_to_copy));
	    }

	    ForwardPropagate(batch_data_placeholder);
	    NNLayer *last = layers[layers.size()-1];
	    double *predictions = last->Output();
	    for (int example = 0; example < n_to_copy; example++) {
		int prediction = Argmax(&predictions[example*last->Dimension()], last->Dimension());
		int truth = Argmax(&batch_labels_placeholder[example*last->Dimension()], last->Dimension());
		if (prediction != truth) n_wrong++;
		n_seen++;
	    }
	}
	free(batch_data_placeholder);
	free(batch_labels_placeholder);
	return n_wrong / n_seen;
    }

    ~NN() {
	for (int i = 0; i < layers.size(); i++) {
	    delete layers[i];
	}
    }

 private:
    std::vector<NNLayer *> layers;
    int batchsize;
    double learning_rate;

    double ComputeBatchLoss(double *data, double *labels, int n_examples) {
	assert(n_examples <= batchsize);
	ForwardPropagate(data);
	NNLayer *last = layers[layers.size()-1];
	double *predictions = last->Output();
	double loss = 0;
	for (int i = 0; i < n_examples; i++) {
	    loss += LogDot(&predictions[i*last->Dimension()],
			   &labels[i*last->Dimension()],
			   last->Dimension());
	}
	return loss;
    }

    void ForwardPropagate(double *data) {
	layers[0]->ForwardPropagate(data);
    }

    void BackPropagate(double *labels) {
	layers[layers.size()-1]->BackPropagate(labels);
    }
};

void test_nn() {

    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(10);

    std::cout << "Test nn..." << std::endl;

    NNParams *params = new NNParams();
    int batch_size = 128;
    params->AddLayer(batch_size, IMAGE_X*IMAGE_Y);
    params->AddLayer(IMAGE_X*IMAGE_Y, 100);
    params->AddLayer(100, 100);
    params->AddLayer(100, N_CLASSES);
    params->SetBatchsize(batch_size);
    params->SetLearningRate(.01);
    NN *nn = new NN(params);
    int number_of_images, number_of_test_images, image_size;
    int number_of_labels, number_of_test_labels;
    uchar **images = read_mnist_images(TRAINING_IMAGES, number_of_images, image_size);
    uchar *labels = read_mnist_labels(TRAINING_LABELS, number_of_labels);
    uchar **test_images = read_mnist_images(TEST_IMAGES, number_of_test_images, image_size);
    uchar *test_labels = read_mnist_labels(TEST_LABELS, number_of_test_labels);

    for (int i = 0; i < 20; i++) {
	if (i % 10 == 0) {
	    double loss = nn->ComputeLoss(images, labels, number_of_images);
	    double err_rate = nn->ComputeErrorRate(images, labels, number_of_images);
	    double test_err_rate = nn->ComputeErrorRate(test_images, test_labels, number_of_test_images);
	    std::cout << "Loss: " << loss << std::endl;
	    std::cout << "Train Error rate: " << err_rate << std::endl;
	    std::cout << "Test Error rate: " << test_err_rate << std::endl;
	}
	nn->Train(images, labels, number_of_images);
    }

    // Final results
    double loss = nn->ComputeLoss(images, labels, number_of_images);
    double err_rate = nn->ComputeErrorRate(images, labels, number_of_images);
    double test_err_rate = nn->ComputeErrorRate(test_images, test_labels, number_of_test_images);
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Train Error rate: " << err_rate << std::endl;
    std::cout << "Test Error rate: " << test_err_rate << std::endl;

    delete nn;
    delete params;

    std::cout << "Test succeeded!" << std::endl;
}

#endif
