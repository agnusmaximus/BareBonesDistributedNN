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

template <typename T>
struct sync_queue {
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
};

//////////////////////////////////
// Thread safe queue helpers    //
//////////////////////////////////

template <typename T>
sync_queue<T> * new_sync_queue() {
    sync_queue<T> *q = new sync_queue<T>();
    return q;
}

template <typename T>
void push_thread_safe(sync_queue<T> *q, T element) {
    std::unique_lock<std::mutex> lock(q->mtx);
    q->q.push(element);
    lock.unlock();
    q->cv.notify_all();
}

template <typename T>
T pop_thread_safe(sync_queue<T> *q) {
    // Blocks on 0 elements.
    std::unique_lock<std::mutex> lock(q->mtx);
    while (q->q.empty()) {
	q->cv.wait(lock);
    }
    T returnvalue = q->q.front();
    q->q.pop();
    return returnvalue;
}

///////////////////////////////////
// Helpers for sending gradients //
///////////////////////////////////
struct GradientSendRequest {
    double *gradient;
    int layer;
    int step;
};

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
