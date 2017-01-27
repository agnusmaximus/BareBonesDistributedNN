#include <iostream>
#include <vector>

using namespace std;

struct nn_params {
    vector<pair<int, int> > layers;

    nn_params() : layers(vector<pair<int, int> >()) {}
};

typedef struct nn_params nn_params;

nn_params * new_nn_params() {
    nn_params *params = new nn_params;
    return params;
}

void free_nn_params(nn_params *params) {
    delete params;
}

void add_layer(nn_params *params, int in_layer_dim, int out_layer_dim) {
    params->layers.push_back(make_pair(in_layer_dim, out_layer_dim));
}

void layer_input_dimension_wrong(nn_params *params, int index, int expected) {
    std::cout << "Input dimension for nn is " << params->layers[index].first << " expected: " << expected << std::endl;
    exit(-1);
}

void layer_output_dimension_wrong(nn_params *params, int index, int expected) {
    std::cout << "Output dimension for nn is " << params->layers[index].second << " expected: " << expected << std::endl;
    exit(-1);
}

void validate_nn_params(nn_params *params, int in_size, int out_size) {
    if (params->layers[0].first != in_size) {
	layer_input_dimension_wrong(params, 0, in_size);
    }
    if (params->layers[params->layers.size()-1].second != out_size) {
	layer_output_dimension_wrong(params, params->layers.size()-1, out_size);
    }
    if (params->layers.size() <= 1) {
	std::cout << "Need more than 1 layer in nn." << std::endl;
	exit(-1);
    }
    int previous_layer_output_dimension = params->layers[0].second;
    for (int i = 1; i < params->layers.size(); i++) {
	if (params->layers[i].first != previous_layer_output_dimension) {
	    layer_input_dimension_wrong(params, i, previous_layer_output_dimension);
	}
	previous_layer_output_dimension = params->layers[i].second;
    }
}

void nn_test() {
    nn_params *params = new_nn_params();
    add_layer(params, 128, 300);
    add_layer(params, 300, 400);
    add_layer(params, 400, 10);
    validate_nn_params(params, 128, 10);
    free_nn_params(params);
}
