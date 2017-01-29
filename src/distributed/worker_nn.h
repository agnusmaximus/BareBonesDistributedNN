#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

class WorkerNN : public NN {
 public:
    WorkerNN(NNParams *params, int rank, int n_procs) : NN(params) {
	this->rank = rank;
	this->n_procs = n_procs;
	this->cur_step = STEP_START;
	this->comm = MPI_COMM_WORLD;

	for (int i = 0; i < layers.size(); i++) {
	    layer_cur_step.push_back(STEP_UNINITIALIZED);
	}
	layer_requests.resize(layers.size());
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {
	// Worker distributed training. Fetch all layer weights.
	AsynchronousFetchWeights();
    }

 protected:

    // The synchronized step (should be the same across workers & master)
    int cur_step, rank, n_procs;
    MPI_Comm comm;

    // layer_cur_step[i] is the iteration step for the current weights
    std::vector<int> layer_cur_step;
    // Requests for fetching each layer.
    std::vector<MPI_Request> layer_requests;

    // Fetch all layer weights asynchronously. (from master).
    void AsynchronousFetchWeights() {

	// Last layer has no weights.
	for (int i = 0; i < layers.size()-1; i++) {
	    // Check if we have already fetched the weights for this
	    // particular step. If so, don't fetch it.
	    std::cout << layers[i]->GetLayerCount() << std::endl;
	    if (layer_cur_step[i] < cur_step) {
		MPI_Irecv(layers[i]->GetLayer(),
			  layers[i]->GetLayerCount(),
			  MPI_DOUBLE,
			  MASTER_RANK,
			  cur_step,
			  this->comm,
			  &layer_requests[i]);
	    }
	}
    }
};

#endif
