#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

class WorkerNN : public NN {
public:
    WorkerNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs) : NN(params), layer_comms(layer_comms) {
	this->rank = rank;

	// Create a memory pool for layer weights
	for (int i = 0; i < layers.size()-1; i++) {
	    memory_pool.push_back(new_sync_queue<double *>());
	    for (int j = 0; j < POOL_SIZE; j++) {
		push_thread_safe(memory_pool[i], (double *)malloc(sizeof(double) * layers[i]->GetLayerCount()));
	    }
	}

	// Memory for layers received
	for (int i = 0; i < layers.size()-1; i++) {
	    layer_weights_recved.push_back(std::vector<sync_queue<double *> *>());
	    for (int j = 0; j < N_TRAIN_ITERS+1; j++) {
		layer_weights_recved[i].push_back(new_sync_queue<double *>());
	    }
	}

	// Gradients send queue
	gradients_to_send = new_sync_queue<GradientSendRequest>();

	// Set local and master step
	local_step = master_step = STEP_UNINITIALIZED;

	// Launch thread for updating the local step
	local_step_update_thread = std::thread(&WorkerNN::UpdateLocalStepAsync, this);

	// Launch thread for sending the gradients
	send_gradients_thread = std::thread(&WorkerNN::SendGradientsAsync, this);

	// Launch thread for receiving the layer weights
	weights_update_thread = std::thread(&WorkerNN::UpdateWeightsAsync, this);
    }

    ~WorkerNN() {

	// Free sync queues for gradients sending.
	free(gradients_to_send);


	// Clear all gradients received
	while (layer_weights_recved.size() != 0) {
	    while (layer_weights_recved.back().size() != 0) {
		free(layer_weights_recved.back().back());
		layer_weights_recved.back().pop_back();
	    }
	    layer_weights_recved.pop_back();
	}

	// Assume all memory elements back in memory queue.
	while (memory_pool.size() != 0) {
	    sync_queue<double *> *q = memory_pool.back();
	    while (!q->q.empty()) {
		double *mem = pop_thread_safe<double *>(q);
		free(mem);
	    }
	    free(q);
	    memory_pool.pop_back();
	}
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {

	while (true) {
	    if (LocalStepDifferentFromMasterStep()) {
		UpdateLocalStep();
		FillNextBatch(data, labels, n_examples);
		std::cout << "WORKER " << rank << " ON " << local_step << std::endl;

		// Forward propagation
		for (int i = 0; i < layers.size(); i++) {
		    if (i != layers.size()-1) {

			// Wait to receive layers of the current step.
			double *weights = pop_thread_safe(layer_weights_recved[i][local_step]);
			memcpy(layers[i]->GetLayer(), weights, sizeof(double) * layers[i]->GetLayerCount());

			// Done with this memory
			push_thread_safe(memory_pool[i], weights);
		    }
		    layers[i]->ForwardPropagateCore(batch_data_placeholder);
		}

		// Backpropagation
		for (int i = layers.size()-1; i >= 0; i--) {
		    layers[i]->BackPropagateCore(batch_labels_placeholder);

		    if (i != layers.size()-1) {

			// Push onto queue for sending gradients
			double *mem = pop_thread_safe(memory_pool[i]);
			memcpy(mem, layers[i]->GetGradient(), sizeof(double) * layers[i]->GetLayerCount());

			push_thread_safe<GradientSendRequest>(gradients_to_send, (GradientSendRequest){mem, i, local_step});
		    }
		}
	    }
	}
    }

protected:

    std::vector<MPI_Comm> layer_comms;
    std::vector<sync_queue<double *> *> memory_pool;
    std::vector<std::vector<sync_queue<double *> *> > layer_weights_recved;
    sync_queue<GradientSendRequest> *gradients_to_send;
    std::thread local_step_update_thread, send_gradients_thread, weights_update_thread;
    int local_step, master_step, rank;

    void UpdateLocalStep() {
	local_step = master_step;
    }

    void UpdateWeightsAsync() {

	// Start receiving for any of the comms.
	std::vector<MPI_Request> requests(layers.size()-1);
	std::vector<double *> memory(layers.size()-1);
	for (int i = 0; i < layers.size()-1; i++) {

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
	    std::cout << "Worker " << rank << " received layer " << layer_received << " for step " << stat.MPI_TAG << std::endl;

	    // Add the received memory to the queue of received layer weights
	    push_thread_safe(layer_weights_recved[layer_received][stat.MPI_TAG], mem);

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
	while (1) {
	    MPI_Recv(&master_step, 1, MPI_INT, MASTER_RANK, STEP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }

    void SendGradientsAsync() {
	while (1) {
	    // Dequeue send requests
	    GradientSendRequest r = pop_thread_safe<GradientSendRequest>(gradients_to_send);
	    std::cout << "Worker " << rank << " Sending gradient... " << r.layer << " " << r.step << std::endl;

	    // Send
	    MPI_Send(r.gradient, layers[r.layer]->GetLayerCount(), MPI_DOUBLE,
		     MASTER_RANK, r.step, layer_comms[r.layer]);

	    // Put memory back into the pool
	    push_thread_safe(memory_pool[r.layer], r.gradient);
	}
    }
};

#endif
