#include <CL/opencl.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>  // Добавлен для std::setprecision

/**
 * @brief Утилиты для работы с OpenCL
 */
class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();
    
    cl_context getContext() const { return context_; }
    cl_command_queue getQueue() const { return queue_; }
    cl_device_id getDevice() const { return device_; }
    
    cl_program createProgramFromSource(const std::string& source);
    void checkError(cl_int err, const std::string& operation);
    void printDeviceInfo() const;
    
private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
};

std::string readKernelFile(const std::string& filename);
