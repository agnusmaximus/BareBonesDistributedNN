#include <iostream>
#include <mpi.h>
#include "distributed/distributed_defines.h"
#include "distributed/worker_nn.h"

int main(void) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Set NN Params
    NNParams *params = new NNParams();
    int batch_size = 128;
    params->AddLayer(batch_size, IMAGE_X*IMAGE_Y);
    params->AddLayer(IMAGE_X*IMAGE_Y, 100);
    params->AddLayer(100, 100);
    params->AddLayer(100, N_CLASSES);
    params->SetBatchsize(batch_size);
    params->SetLearningRate(.01);


    if (rank == MASTER_RANK) {

    }
    else {
	WorkerNN *worker = new WorkerNN(params, rank, n_procs);
	delete worker;
    }

    delete params;

    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize the MPI environment.
    MPI_Finalize();
}
