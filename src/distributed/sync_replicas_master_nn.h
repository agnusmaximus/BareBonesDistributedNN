#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

class SyncReplicasMasterNN : public NN {
 public:

    SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs, int n_to_collect) : NN(params), layer_comms(layer_comms) {
	this->n_procs = n_procs;
	master_step = STEP_START;

	// Initialize request variables for broadcasting steps and weights.
	for (int i = 0; i < n_procs; i++) {
	    broadcast_master_step_requests.push_back(MPI_REQUEST_NULL);
	    broadcast_weight_requests.push_back(std::vector<MPI_Request>());
	    for (int layer = 0; layer < layers.size(); layer++) {
		broadcast_weight_requests[i].push_back(MPI_REQUEST_NULL);
	    }
	}

	// Create a memory pool
	for (int i = 0; i < layers.size()-1; i++) {
	    memory_pool.push_back(new_sync_queue<double *>());
	    for (int j = 0; j < POOL_SIZE; j++) {
		push_thread_safe(memory_pool[i], (double *)malloc(sizeof(double) * layers[i]->GetLayerCount()));
	    }
	}

	// Memory for Gradients received
	for (int i = 0; i < layers.size()-1; i++) {
	    gradients_recved.push_back(std::vector<sync_queue<double *> *>());
	    for (int j = 0; j < N_TRAIN_ITERS+1; j++) {
		gradients_recved[i].push_back(new_sync_queue<double *>());
	    }
	}

	// Thread for receiving gradients.
	receive_gradients_thread = std::thread(&SyncReplicasMasterNN::ReceiveGradientsAsync, this);
    }

    ~SyncReplicasMasterNN() {

	// Clear all gradients received
	while (gradients_recved.size() != 0) {
	    while (gradients_recved.back().size() != 0) {
		free(gradients_recved.back().back());
		gradients_recved.back().pop_back();
	    }
	    gradients_recved.pop_back();
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

    void Train(uchar **data, uchar *labels, int examples) override {
	while (true) {
	    BroadcastMasterStep();
	    BroadcastWeights();


	    master_step++;
	    break;
	}
    }

protected:
    std::thread receive_gradients_thread;

    std::vector<MPI_Comm> layer_comms;
    std::vector<MPI_Request> broadcast_master_step_requests;
    std::vector<std::vector<MPI_Request>> broadcast_weight_requests;
    int master_step, master_step_copy;
    int n_procs;

    // Memory pool for receiving gradients.
    std::vector<sync_queue<double *> *> memory_pool;

    // Queue for received gradients
    std::vector<std::vector<sync_queue<double *> *> > gradients_recved;

    // Sends master step to all workers
    void BroadcastMasterStep() {
	master_step_copy = master_step;
	for (int i = 0; i < n_procs; i++) {
	    if (i != MASTER_RANK) {
		if (broadcast_master_step_requests[i] != MPI_REQUEST_NULL) {
		    MPI_Wait(&broadcast_master_step_requests[i], MPI_STATUS_IGNORE);
		}
		MPI_Isend(&master_step, 1, MPI_INT, i, STEP_TAG, MPI_COMM_WORLD, &broadcast_master_step_requests[i]);
	    }
	}
    }

    // Sends weights to all workers
    void BroadcastWeights() {
	master_step_copy = master_step;
	for (int layer = 0; layer < layers.size()-1; layer++) {
	    for (int i = 0; i < n_procs; i++) {
		if (i != MASTER_RANK) {
		    if (broadcast_weight_requests[i][layer] != MPI_REQUEST_NULL) {
			MPI_Wait(&broadcast_weight_requests[i][layer], MPI_STATUS_IGNORE);
		    }
		    MPI_Isend(layers[layer]->GetLayer(), layers[layer]->GetLayerCount(), MPI_DOUBLE, i, master_step, layer_comms[layer], &broadcast_weight_requests[i][layer]);
		}
	    }
	}
    }

    // Thread function to continually receive gradients
    void ReceiveGradientsAsync() {

	// Initate requests for all layers.
	std::vector<MPI_Request> requests(layers.size()-1);
	std::vector<double *> memory(layers.size()-1);
	for (int i = 0; i < layers.size()-1; i++) {
	    double *mem = pop_thread_safe(memory_pool[i]);
	    memory[i] = mem;
	    MPI_Irecv(mem, layers[i]->GetLayerCount(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, layer_comms[i], &requests[i]);
	}

	while (true) {

	    // Wait for any request to be satisfied.
	    int layer_received = -1;
	    MPI_Status stat;
	    MPI_Waitany(layers.size()-1, requests.data(), &layer_received, &stat);
	    double *mem = memory[layer_received];

	    std::cout << "Master received gradient for layer " << layer_received << std::endl;

	    // Add the gradient to the queue
	    push_thread_safe(gradients_recved[layer_received][stat.MPI_TAG], mem);

	    // Initiate a new request in place of the satisfied request.
	    mem = pop_thread_safe(memory_pool[layer_received]);
	    memory[layer_received] = mem;
	    MPI_Irecv(mem, layers[layer_received]->GetLayerCount(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, layer_comms[layer_received], &requests[layer_received]);
	}
    }
};

#endif
