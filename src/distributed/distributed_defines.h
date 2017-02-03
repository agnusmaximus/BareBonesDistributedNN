#ifndef _distributed_defines_
#define _distributed_defines_

#include <thread>
#include <queue>
#include <mutex>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <time.h>
#include <cassert>
#include "../mnist/mnist.h"
#include "../nn/nn.h"
#include "../nn/nn_params.h"
#include "../nn/nn_layer.h"
#include <condition_variable>
#include <mpi.h>

#define STEP_TAG 0
#define STEP_START 1
#define STEP_UNINITIALIZED (STEP_START-1)
#define MASTER_RANK 0
#define EVALUATOR_RANK 1
#define POOL_SIZE 100
#ifndef SHORTCIRCUIT
#define SHORTCIRCUIT true
#endif
#define GENERATE_TIMELINE false
#define N_TRAIN_ITERS 100

struct sync_queue {
    std::queue<double *> q;
    std::mutex mtx;
    std::condition_variable cv;
};
typedef sync_queue sync_queue;

//////////////////////////////////
// Thread safe queue helpers    //
//////////////////////////////////

sync_queue * new_sync_queue() {
    sync_queue *q = new sync_queue();
    return q;
}

void push_thread_safe(sync_queue *q, double *element) {
    std::unique_lock<std::mutex> lock(q->mtx);
    q->q.push(element);
    lock.unlock();
    q->cv.notify_all();
}

double * pop_thread_safe(sync_queue *q) {
    // Blocks on 0 elements.
    std::unique_lock<std::mutex> lock(q->mtx);
    while (q->q.empty()) {
	q->cv.wait(lock);
    }
    double *returnvalue = q->q.front();
    q->q.pop();
    return returnvalue;
}


string scheme_full_name(string scheme_name, int n_to_collect, int n_procs) {

    // -2 for master and evaluator
    string name = scheme_name + std::to_string(n_to_collect) + "_"  + std::to_string(n_procs-2);
    if (SHORTCIRCUIT) {
	name += "_shortcircuit";
    }
    else {
	name += "_no_shortcircuit";
    }
    return name;
}

#endif
