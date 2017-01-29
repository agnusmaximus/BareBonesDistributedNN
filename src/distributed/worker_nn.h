#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

class WorkerNN : public NN {
 public:
    WorkerNN(NNParams *params) : NN(params) {

    }
};

#endif
