#include <iostream>
#include <vector>

using namespace std;

class ParamsNN {

 public:

    ParamsNN() {
    }

    ~ParamsNN() {
    }

    void AddLayer(int in_layer_dim, int out_layer_dim) {
	layers.push_back(make_pair(in_layer_dim, out_layer_dim));
    }

    void Validate(int in_size, int out_size) {
	if (layers[0].first != in_size) {
	    LayerInputDimensionWrong(0, in_size);
	}
	if (layers[layers.size()-1].second != out_size) {
	    LayerOutputDimensionWrong(layers.size()-1, out_size);
	}
	if (layers.size() <= 1) {
	    std::cout << "Need more than 1 layer in nn." << std::endl;
	    exit(-1);
	}

	int previous_layer_output_dimension = layers[0].second;
	for (int i = 1; i < layers.size(); i++) {
	    if (layers[i].first != previous_layer_output_dimension) {
		LayerInputDimensionWrong(i, previous_layer_output_dimension);
	    }
	    previous_layer_output_dimension = layers[i].second;
	}
    }

 private:

    std::vector<pair<int, int> > layers;


    void LayerInputDimensionWrong(int index, int expected) {
	std::cout << "Input dimension for nn is " << layers[index].first << " expected: " << expected << std::endl;
	exit(-1);
    }

    void LayerOutputDimensionWrong(int index, int expected) {
	std::cout << "Output dimension for nn is " << layers[index].second << " expected: " << expected << std::endl;
	exit(-1);
    }

};

void nn_test() {

    ParamsNN *params = new ParamsNN();
    params->AddLayer(128, 300);
    params->AddLayer(300, 400);
    params->AddLayer(400, 10);
    params->Validate(128, 10);
    delete params;
}
