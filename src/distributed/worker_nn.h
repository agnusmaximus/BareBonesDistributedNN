#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

class WorkerNN : public NN {
public:
    WorkerNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs) : NN(params), layer_comms(layer_comms) {

	// Launch thread for updating the local step
	local_step_update_thread = std::thread(&WorkerNN::UpdateLocalStepAsync, this);

	// Launch thread for sending the gradients
	send_gradients_thread = std::thread(&WorkerNN::SendGradientsAsync, this);

	// Launch thread for receiving the layer weights
	weights_update_thread = std::thread(&WorkerNN::UpdateWeightsAsync, this);

	// Create a memory pool for layer weights
	for (int i = 0; i < layers.size()-1; i++) {
	    memory_pool.push_back(new_sync_queue());
	    for (int j = 0; j < POOL_SIZE; j++) {
		push_thread_safe(memory_pool[i], (double *)malloc(sizeof(double) * layers[i]->GetLayerCount()));
	    }
	}

	// Memory for layers received
	for (int i = 0; i < layers.size()-1; i++) {
	    layer_weights.push_back(std::vector<sync_queue *>());
	    for (int j = 0; j < N_TRAIN_ITERS+1; j++) {
		layer_weights[i].push_back(new_sync_queue());
	    }
	}

	// Set local and master step
	local_step = master_step = STEP_UNINITIALIZED;
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {

	while (true) {
	    if (LocalStepDifferentFromMasterStep()) {
		UpdateLocalStep();
		FillNextBatch(data, labels, n_examples);
		std::cout << local_step << std::endl;

		// Forward propagation
		for (int i = 0; i < layers.size(); i++) {
		    if (i != layers.size()-1) {
			// Wait to receive layers of the current step.
			double *weights = pop_thread_safe(layer_weights[i][local_step]);
			memcpy(layers[i]->GetLayer(), weights, sizeof(double) * layers[i]->GetLayerCount());

			// Done with this memory
			push_thread_safe(memory_pool[i], weights);
		    }
		    layers[i]->ForwardPropagateCore(batch_data_placeholder);
		}

		// Backpropagation
		for (int i = layers.size(); i >= 0; i--) {
		    layers[i]->BackPropagateCore(batch_labels_placeholder);

		    if (i != layers.size()-1) {

			// Push onto queue for sending gradients
			double *mem = pop_thread_safe(memory_pool[i]);
			memcpy(mem, layers[i]->GetGradient(), sizeof(double) * layers[i]->GetLayerCount());

		    }
		}
	    }
	}
    }

protected:

    std::vector<MPI_Comm> layer_comms;
    std::vector<sync_queue *> memory_pool;
    std::vector<std::vector<sync_queue *> > layer_weights;
    std::thread local_step_update_thread, send_gradients_thread, weights_update_thread;
    int local_step, master_step;

    void UpdateLocalStep() {
	local_step = master_step;
    }

    void UpdateWeightsAsync() {

	// Start receiving for any of the comms.
	std::vector<MPI_Request> requests(layers.size()-1);
	std::vector<double *> memory;
	for (int i = 0; i < layers.size(); i++) {

	    // Get a layer from the memory pool.
	    double *mem = pop_thread_safe(memory_pool[i]);
	    memory[i] = mem;

	    // Try to receive weight updates from master
	    MPI_Irecv(mem, layers[i]->GetLayerCount(), MPI_DOUBLE, MASTER_RANK, MPI_ANY_TAG, layer_comms[i], &requests[i]);
	}

	while (true) {

	    // Wait for any layers
	    int layer_received = -1;
	    MPI_Status stat;
	    MPI_Waitany(layers.size()-1, requests.data(), &layer_received, &stat);
	    double *mem = memory[layer_received];

	    // Add the received memory to the queue of received layer weights
	    push_thread_safe(layer_weights[layer_received][stat.MPI_TAG], mem);

	    // Initiate another request for weights of the same layer
	    mem = pop_thread_safe(memory_pool[layer_received]);
	    memory[layer_received] = mem;
	    MPI_Irecv(mem, layers[layer_received]->GetLayerCount(), MPI_DOUBLE, MASTER_RANK, MPI_ANY_TAG, layer_comms[layer_received], &requests[layer_received]);
	}
    }

    bool LocalStepDifferentFromMasterStep() {
	return local_step != master_step;
    }

    void UpdateLocalStepAsync() {
	MPI_Recv(&master_step, 1, MPI_INT, MASTER_RANK, STEP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void SendGradientsAsync() {

    }
};

#endif
