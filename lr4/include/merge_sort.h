#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include "gpu_helper.h"
#include <vector>

class MergeSorter {
public:
    MergeSorter();
    ~MergeSorter();
    
    std::vector<float> sort(const std::vector<float>& data, size_t local_size = 0);
    static std::vector<float> sort_cpu(const std::vector<float>& data);
    static bool verify(const std::vector<float>& data);
    
private:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_program program;
};

#endif
