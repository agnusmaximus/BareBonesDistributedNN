#ifndef _EVALUATOR_NN_
#define _EVALUATOR_NN_

#include "distributed_defines.h"

class EvaluatorNN : public WorkerNN {
public:
    EvaluatorNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs) : WorkerNN(params, layer_comms, rank, n_procs) {

    }

    ~EvaluatorNN() {

    }

    void Train(uchar **data, uchar *labels, int n_examples) override {
	double start_time = GetTimeMillis();
	while (true) {
	    if (LocalStepDifferentFromMasterStep()) {
		UpdateLocalStep();
		for (int i = 0; i < layers.size(); i++) {
		    if (i != layers.size()-1) {
			double *weights = pop_thread_safe(layer_weights_recved[i][local_step]);
			memcpy(layers[i]->GetLayer(), weights, sizeof(double) * layers[i]->GetLayerCount());
			push_thread_safe(memory_pool[i], weights);
		    }
		}
		double loss = ComputeLoss(data, labels, n_examples);
		double err = ComputeErrorRate(data, labels, n_examples);
		double t_elapsed = GetTimeMillis() - start_time;
		std::cout << t_elapsed << " " << loss << " " << err << std::endl;
	    }
	    else {
		sleep(0);
	    }
	}
    }

};

#endif
