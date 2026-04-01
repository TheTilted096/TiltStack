#include "Scheduler.h"

class Orchestrator {
  public:
    std::vector<std::thread> threadPool;
    InferenceQueue iq;

    Orchestrator();
    ~Orchestrator();

    void runWorker();

    void submitBatch();

    void initThreadPool();

    void runIteration();
};