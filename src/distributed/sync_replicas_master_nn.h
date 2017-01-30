#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

class SyncReplicasMasterNN : public NN {
 public:
   SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs, int n_to_collect) : NN(params), layer_comms(layer_comms) {
	this->comm = MPI_COMM_WORLD;
	this->n_to_collect = n_to_collect;
	this->n_procs = n_procs;
	this->cur_step = STEP_START;
	layer_send_requests.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
	    for (int j = 0; j < n_procs; j++) {
		layer_send_requests[i].push_back(MPI_REQUEST_NULL);
	    }
	}

	// Preallocate memory for gradient buffers for irecv.
	for (int i = 0; i < layers.size()-1; i++) {
	    grad_buffers.push_back(std::vector<double *>());
	    for (int j = 0; j < N_RECV_REQUESTS_PER_LAYER; j++) {
		grad_buffers[i].push_back((double *)malloc(sizeof(double) * layers[i]->GetLayerCount()));
	    }
	}
    }

    void Train(uchar **data, uchar *labels, int examples) override {

	std::vector<int> gradients_accumulated(layers.size());
	std::fill(gradients_accumulated.begin(),
		  gradients_accumulated.end(),
		  0);

	AsynchronousFetchGradientsStart();

	double time_start = GetTimeMillis();

	while (true) {
	    AsynchronousBroadcastStep();
	    AsynchronousBroadcastLayerWeights();

	    bool enough_gradients_received = false;
	    while (!enough_gradients_received) {

		// While we don't have enough gradients, keep waiting to receive them.
		int index_received = -1;
		MPI_Status stat;
		MPI_Waitany((layers.size()-1) * N_RECV_REQUESTS_PER_LAYER,
			    gradient_fetch_requests.data(),
			    &index_received,
			    &stat);

		// We push N_RECV_REQUESTS_PER_LAYER per layer. The layer is
		// index / N_RECV_REQUESTS_PER_LAYER
		int layer_received = index_received / N_RECV_REQUESTS_PER_LAYER;
		int copy_index = index_received - layer_received * N_RECV_REQUESTS_PER_LAYER;

		if (stat.MPI_TAG == cur_step) {

		    int count = 0;
		    MPI_Get_count(&stat, MPI_DOUBLE, &count);
		    assert(count == layers[layer_received]->GetLayerCount());

		    if (gradients_accumulated[layer_received] < n_to_collect) {
			gradients_accumulated[layer_received]++;

			// Apply gradient
			layers[layer_received]->ApplyGrad(learning_rate, grad_buffers[layer_received][copy_index]);

			enough_gradients_received = true;
			for (int i = 0; i < layers.size()-1; i++) {
			    enough_gradients_received = enough_gradients_received && gradients_accumulated[i] >= n_to_collect;
			}
		    }
		}

		// Received a gradient for this layer... Initiate a new
		// request to receive another one at the layer for the specific copy index.
		AsynchronousFetchGradient(layer_received, copy_index, &gradient_fetch_requests[index_received]);
	    }

	    std::fill(gradients_accumulated.begin(),
		      gradients_accumulated.end(), 0);

	    if (cur_step % 50 == 0) {
		double err_rate = ComputeErrorRate(data, labels, examples);
		double loss = ComputeLoss(data, labels, examples);
		double time = GetTimeMillis() - time_start;
		std::cout << time << " - STEP: " << cur_step << " ERROR: " << err_rate << " LOSS: " << loss << std::endl;
	    }

	    cur_step++;
	}
    }

 protected:
    MPI_Request step_broadcast_req;
    int n_procs, cur_step, n_to_collect;
    MPI_Comm comm;
    std::vector<std::vector<MPI_Request> > layer_send_requests;
    std::vector<MPI_Request> gradient_fetch_requests;
    std::vector<MPI_Comm> &layer_comms;
    std::vector<std::vector<double *> > grad_buffers;

    void AsynchronousBroadcastStep() {
	for (int i = 0; i < n_procs; i++) {
	    if (i != MASTER_RANK) {
		MPI_Isend(&cur_step, 1, MPI_INT, i, STEP_TAG, comm, &step_broadcast_req);
	    }
	}
    }

    void AsynchronousFetchGradient(int l, int copy, MPI_Request *req) {
	MPI_Irecv(grad_buffers[l][copy],
		  layers[l]->GetLayerCount(),
		  MPI_DOUBLE,
		  MPI_ANY_SOURCE,
		  MPI_ANY_TAG,    // Any gradient from any iteration may be fetched.
		  layer_comms[l],
		  req);
    }

    void AsynchronousFetchGradientsStart() {
	for (int i = 0; i < layers.size()-1; i++) {
	    for (int k = 0; k < N_RECV_REQUESTS_PER_LAYER; k++) {
		gradient_fetch_requests.push_back(MPI_REQUEST_NULL);
	    }
	}

	// We initiate a ton of requests to receive gradients for each layer.
	for (int l = 0; l < layers.size()-1; l++) {
	    for (int k = 0; k < N_RECV_REQUESTS_PER_LAYER; k++) {
		AsynchronousFetchGradient(l, k, &gradient_fetch_requests[l*N_RECV_REQUESTS_PER_LAYER+k]);
	    }
	}
    }

    void AsynchronousBroadcastLayerWeights() {
	for (int l = 0; l < layers.size()-1; l++) {
	    for (int i = 0; i < n_procs; i++) {
		if (i != MASTER_RANK) {
		    if (layer_send_requests[l][i] != MPI_REQUEST_NULL) {
			MPI_Request_free(&layer_send_requests[l][i]);
		    }

		    MPI_Isend(layers[l]->GetLayer(),
			      layers[l]->GetLayerCount(),
			      MPI_DOUBLE,
			      i,
			      cur_step,
			      layer_comms[l],
			      &layer_send_requests[l][i]);
		}
	    }
	}
    }
};

#endif
