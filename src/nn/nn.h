#include <iostream>
#include <vector>
#include "nn_params.h"
#include "nn_layer.h"

class NN {
 public:
    NN(const NNParams &params) {

    }

    ~NN() {

    }
};

void test_nn() {

    std::cout << "Testing nn..." << std::endl;

    NNParams *params = new NNParams();
    params->AddLayer(128, 300);
    params->AddLayer(300, 400);
    params->AddLayer(400, 10);
    params->Validate(128, 10);

    NN *nn = new NN(*params);

    delete nn;
    delete params;

    std::cout << "Done testing nn!" << std::endl;
}
