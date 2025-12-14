#ifndef GPU_SORTER_HPP
#define GPU_SORTER_HPP

#include <vector>
#include <CL/opencl.hpp>

struct GPUConfig {
    size_t work_group_size = 256;  // GROUP_SIZE из kernel
};

class GPUMergeSorter {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel merge_kernel;
    cl::Device device;
    
public:
    GPUMergeSorter();
    void sort(std::vector<int>& array, const GPUConfig& config = GPUConfig());
    double sort_with_profiling(std::vector<int>& array, const GPUConfig& config = GPUConfig());
    
    size_t get_max_work_group_size() const;
    size_t get_preferred_work_group_size() const;
    std::string get_device_info() const;
};

#endif