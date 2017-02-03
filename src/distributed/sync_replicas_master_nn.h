#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

class SyncReplicasMasterNN : public NN {
 public:

    SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs, int n_to_collect) : NN(params), layer_comms(layer_comms) {
    }

    ~SyncReplicasMasterNN() {
    }

    void Train(uchar **data, uchar *labels, int examples) override {
    }

protected:
    std::vector<MPI_Comm> layer_comms;
};

#endif
