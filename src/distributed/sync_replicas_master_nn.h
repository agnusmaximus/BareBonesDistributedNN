#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

static void testme() {
    while (1) {}
}
static void testme2() {

}
static void testme3() {

}
static void testme4() {

}

class SyncReplicasMasterNN : public NN {
 public:

    SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs, int n_to_collect) : NN(params), layer_comms(layer_comms) {
	this->n_procs = n_procs;
	this->n_to_collect = n_to_collect;
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
		double *new_memory = (double *)malloc(sizeof(double) * layers[i]->GetLayerCount());
		if (!new_memory) {
		    std::cout << "Out of memory." << std::endl;
		    exit(0);
		}
		push_thread_safe(memory_pool[i], new_memory);
	    }
	}

	// Memory for Gradients received
	gradients_recved = new_sync_queue<GradientReceiveRequest>();

	// Thread for receiving gradients.
	for (int i = 0; i < N_RECEIVE_THREADS; i++) {
	    receive_gradients_threads.push_back(std::thread(&SyncReplicasMasterNN::ReceiveGradientsAsync, this));
	}

	SetName();
	exchange_names(name, MASTER_RANK);
    }

    ~SyncReplicasMasterNN() {

	free(gradients_recved);

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
	    WaitForAccumulatedGradients();
	    ApplyAllGradients();
	    std::cout << "Master finished iteration " << master_step << std::endl;
	    master_step++;
	}
	while (true) { sleep(0); }
    }

protected:
    std::vector<std::thread> receive_gradients_threads;
    std::string name;

    std::vector<MPI_Comm> layer_comms;
    std::vector<MPI_Request> broadcast_master_step_requests;
    std::vector<std::vector<MPI_Request>> broadcast_weight_requests;
    int master_step, master_step_copy;
    int n_procs, n_to_collect;

    // Memory pool for receiving gradients.
    std::vector<sync_queue<double *> *> memory_pool;

    // Queue for received gradients
    sync_queue<GradientReceiveRequest> *gradients_recved;

    // Sends master step to all workers
    void BroadcastMasterStep() {
	master_step_copy = master_step;
	for (int i = 0; i < n_procs; i++) {
	    if (i != MASTER_RANK) {
		if (broadcast_master_step_requests[i] != MPI_REQUEST_NULL) {
		    MPI_Wait(&broadcast_master_step_requests[i], MPI_STATUS_IGNORE);
		}
		MPI_Isend(&master_step_copy, 1, MPI_INT, i, STEP_TAG, MPI_COMM_WORLD, &broadcast_master_step_requests[i]);
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
			MPI_Request_free(&broadcast_weight_requests[i][layer]);
		    }
		    MPI_Isend(layers[layer]->GetLayer(), layers[layer]->GetLayerCount(), MPI_DOUBLE, i, master_step, layer_comms[layer], &broadcast_weight_requests[i][layer]);
		}
	    }
	}
    }

    void ApplyAllGradients() {
	//std::cout << "Master applying all gradients accumulated..." << std::endl;
	for (int i = 0; i < layers.size()-1; i++) {
	    layers[i]->ApplyGrad(learning_rate, layers[i]->GetGradient());
	}
    }

    bool DoneAccumulating(int *gradients_accumulated) {
	bool done = true;
	for (int i = 0; i < layers.size()-1; i++) {
	    done = done && (gradients_accumulated[i] >= n_to_collect);
	}
	return done;
    }

    void PrintNumAccumulated(int *gradients_accumulated) {
	for (int i = 0; i < layers.size()-1; i++) {
	    std::cout << '[' << gradients_accumulated[i] << ']';
	}
	std::cout << std::endl;
    }

    void WaitForAccumulatedGradients() {

	// Keep track of gradients accumulated.
	int gradients_accumulated[layers.size()-1];
	memset(gradients_accumulated, 0, sizeof(int) * (layers.size()-1));

	// Reset the layers' gradients.
	for (int layer = 0; layer < layers.size()-1; layer++) {
	    memset(layers[layer]->GetGradient(), 0,
		   sizeof(double) * layers[layer]->GetLayerCount());
	}

	while (!DoneAccumulating(gradients_accumulated)) {
	    GradientReceiveRequest r = pop_thread_safe<GradientReceiveRequest>(gradients_recved);

	    // Found a gradient for the current step
	    if (r.step == master_step) {
		gradients_accumulated[r.layer]++;

		// Apply the gradient
		MatrixAdd(r.gradient, layers[r.layer]->GetGradient(), layers[r.layer]->GetGradient(),
			  1, 1,
			  layers[r.layer]->NRows(),
			  layers[r.layer]->NCols(),
			  layers[r.layer]->NCols(),
			  layers[r.layer]->NCols(),
			  layers[r.layer]->NCols());
	    }

	    PrintNumAccumulated(gradients_accumulated);

	    // Return memory back to pool.
	    push_thread_safe<double *>(memory_pool[r.layer], r.gradient);
	}
    }

    // Thread function to continually receive gradients
    void ReceiveGradientsAsync(void) {

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
	    //WaitAnyReverse(requests, &layer_received, &stat);
	    double *mem = memory[layer_received];

	    // Add the gradient to the queue
	    push_thread_safe<GradientReceiveRequest>(gradients_recved, {mem, layer_received, stat.MPI_TAG});

	    // Initiate a new request in place of the satisfied request.
	    mem = pop_thread_safe<double *>(memory_pool[layer_received]);
	    memory[layer_received] = mem;
	    MPI_Irecv(mem, layers[layer_received]->GetLayerCount(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, layer_comms[layer_received], &requests[layer_received]);
	}
    }

    void SetName() {
	name = "NN_" + std::to_string(n_to_collect) + "_" + std::to_string(n_procs-2);
	if (SHORTCIRCUIT) {
	    name += "_shortcircuit";
	}
    }
};

#endif
