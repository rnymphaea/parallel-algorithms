#ifndef SORTER_GPU_HPP
#define SORTER_GPU_HPP

#include <vector>
#include <CL/opencl.h>
#include <string>

struct GpuConfig {
    size_t work_group_size = 256;
};

class SorterGpu {
private:
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel merge_kernel_;
    cl_device_id device_;
    
public:
    SorterGpu();
    ~SorterGpu();
    
    void sort(std::vector<int>& array, const GpuConfig& config = GpuConfig());
    double sortWithProfiling(std::vector<int>& array, const GpuConfig& config = GpuConfig());
    
    size_t getMaxWorkGroupSize() const;
    size_t getPreferredWorkGroupSize() const;
    std::string getDeviceInfo() const;
    
private:
    void checkError(cl_int err, const std::string& operation);
};

#endif
