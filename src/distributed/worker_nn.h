#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

class WorkerNN : public NN {
 public:
   WorkerNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs, bool shortcircuit) : NN(params), layer_comms(layer_comms) {
	this->rank = rank;
	this->shortcircuit = shortcircuit;
	this->n_procs = n_procs;
	this->cur_step = STEP_UNINITIALIZED;
	this->comm = MPI_COMM_WORLD;
	this->next_step = STEP_UNINITIALIZED;

	for (int i = 0; i < layers.size(); i++) {
	    layer_cur_step.push_back(STEP_UNINITIALIZED);
	}
	layer_requests.resize(layers.size());
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {

	SynchronousFetchStep();
	assert(UpdateStep());
	assert(cur_step == STEP_START);

	std::cout << "Worker " << rank << " starting training..." << std::endl;

	while (true) {
	    UpdateStep();
	    AsynchronousFetchWeights();
	    AsynchronousFetchStepUpdate();
	    FillNextBatch(data, labels, n_examples);

	    // Forward propagate
	    for (int i = 0; i < layers.size(); i++) {
		if (shortcircuit && StepChanged()) continue;
		if (i != layers.size()-1) {
		    MPI_Wait(&layer_requests[i], MPI_STATUS_IGNORE);
		    layer_cur_step[i] = cur_step;
		}
		layers[i]->ForwardPropagateCore(batch_data_placeholder);
	    }

	    // Back propagate
	    for (int i = layers.size()-1; i >= 0; i--) {
		if (shortcircuit && StepChanged()) continue;
		layers[i]->BackPropagateCore(batch_labels_placeholder);
		if (i != layers.size()-1) {
		    MPI_Request throwaway;
		    MPI_Isend(layers[i]->GetLayer(),
			      layers[i]->GetLayerCount(),
			      MPI_DOUBLE,
			      MASTER_RANK,
			      cur_step,
			      this->comm,
			      &throwaway);
		}
	    }
	}
    }

 protected:

    // The synchronized step (should be the same across workers & master)
    int cur_step, rank, n_procs, next_step;
    bool shortcircuit;
    MPI_Comm comm;

    // layer_cur_step[i] is the iteration step for the current weights
    std::vector<int> layer_cur_step;

    // Requests for fetching each layer.
    std::vector<MPI_Request> layer_requests;

    // Layer communicator handles
    std::vector<MPI_Comm> &layer_comms;

    // Requests for fetching the step.
    MPI_Request step_fetch_request;

    bool StepChanged() {
	return cur_step != next_step;
    }

    // Returns whether fetched new step was different from cur step.
    bool UpdateStep() {
	int last_step = cur_step;
	cur_step = next_step;
	return last_step != cur_step;
    }

    void AsynchronousFetchStepUpdate() {
	static bool first_fetch = true;
	int completed_step_fetch = 0;
	if (first_fetch)
	    completed_step_fetch = 1;
	else
	    MPI_Test(&step_fetch_request, &completed_step_fetch, MPI_STATUS_IGNORE);
	if (completed_step_fetch) {
	    AsynchronousFetchStep();
	}
	first_fetch = false;
    }

    void AsynchronousFetchStep() {
	MPI_Irecv(&next_step,
		  1,
		  MPI_INT,
		  MASTER_RANK,
		  STEP_TAG,
		  this->comm,
		  &step_fetch_request);
    }

    void SynchronousFetchStep() {
	MPI_Recv(&next_step,
		 1,
		 MPI_INT,
		 MASTER_RANK,
		 STEP_TAG,
		 this->comm,
		 MPI_STATUS_IGNORE);
    }

    // Fetch all layer weights asynchronously. (from master).
    void AsynchronousFetchWeights() {

	// Last layer has no weights.
	for (int i = 0; i < layers.size()-1; i++) {
	    // Check if we have already fetched the weights for this
	    // particular step. If so, don't fetch it.
	    if (layer_cur_step[i] < cur_step) {
		MPI_Irecv(layers[i]->GetLayer(),
			  layers[i]->GetLayerCount(),
			  MPI_DOUBLE,
			  MASTER_RANK,
			  cur_step,
			  layer_comms[i],
			  &layer_requests[i]);
	    }
	}
    }
};

#endif
