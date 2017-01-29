#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

class SyncReplicasMasterNN : public NN {
 public:
   SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs) : NN(params), layer_comms(layer_comms) {
	this->comm = MPI_COMM_WORLD;
	this->n_procs = n_procs;
	this->cur_step = STEP_START;
	layer_send_requests.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
	    layer_send_requests[i].resize(n_procs);
	}
    }

    void Train(uchar **data, uchar *labels, int examples) override {
	while (true) {
	    if (cur_step == STEP_START) {
		AsynchronousBroadcastStep();
		AsynchronousBroadcastLayerWeights();
		return;
	    }

	    cur_step++;
	}
    }

 protected:
    MPI_Request step_broadcast_req;
    int n_procs, cur_step;
    MPI_Comm comm;
    std::vector<std::vector<MPI_Request> > layer_send_requests;
    std::vector<MPI_Comm> &layer_comms;

    void AsynchronousBroadcastStep() {
	for (int i = 0; i < n_procs; i++) {
	    if (i != MASTER_RANK) {
		MPI_Isend(&cur_step, 1, MPI_INT, i, STEP_TAG, comm, &step_broadcast_req);
	    }
	}
    }

    void AsynchronousBroadcastLayerWeights() {
	for (int l = 0; l < layers.size()-1; l++) {
	    for (int i = 0; i < n_procs; i++) {
		if (i != MASTER_RANK) {
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
