#ifndef SORTER_GPU_HPP
#define SORTER_GPU_HPP

#include <vector>
#include <CL/opencl.h>
#include <memory>

struct GPUConfig {
    size_t work_group_size = 256;
};

class SorterGPU {
private:
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel merge_kernel_;
    cl_device_id device_;
    
public:
    SorterGPU();
    ~SorterGPU();
    
    void sort(std::vector<int>& array, const GPUConfig& config = GPUConfig());
    double sortWithProfiling(std::vector<int>& array, const GPUConfig& config = GPUConfig());
    
    size_t getMaxWorkGroupSize() const;
    size_t getPreferredWorkGroupSize() const;
    std::string getDeviceInfo() const;
    
private:
    void checkError(cl_int err, const std::string& operation);
};

#endif
