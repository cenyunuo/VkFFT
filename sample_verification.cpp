//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <complex>
#include <cmath>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#elif(VKFFT_BACKEND==4)
#include <ze_api.h>
#elif(VKFFT_BACKEND==5)
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#endif
#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "sample_verification.h"

VkFFTResult sample_verification_VkFFT_single(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
    VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
    cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
    hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
    cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
    ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif

    if (file_output)
        fprintf(output, "VkFFT Verification Test - Simple FFT with result output\n");
    printf("VkFFT Verification Test - Simple FFT with result output\n");

    // Simple test parameters
    const uint64_t N = 16;  // Small size for easy verification
    const uint64_t numberBatches = 1;

    // Create simple input data
    const uint64_t inputSize = N * 2;  // Complex numbers (real + imag)
    float* buffer_input = (float*)malloc(inputSize * sizeof(float));
    if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;

    // Initialize with simple pattern for easy verification
    for (uint64_t i = 0; i < N; i++) {
        buffer_input[2*i] = (float)i;     // Real part: 0, 1, 2, 3, ...
        buffer_input[2*i+1] = 0.0f;       // Imaginary part: all zeros
    }

    // Print input data
#if (VKFFT_BACKEND==1)
    FILE* result_file = fopen("fft_verification_results_cuda.txt", "w");
#elif (VKFFT_BACKEND==3)
    FILE* result_file = fopen("fft_verification_results_opencl.txt", "w");
#else
    FILE* result_file = fopen("fft_verification_results.txt", "w");
#endif
    if (result_file) {
        fprintf(result_file, "FFT Verification Results\n");
        fprintf(result_file, "========================\n\n");
        fprintf(result_file, "Input data (N=%llu):\n", N);
        for (uint64_t i = 0; i < N; i++) {
            fprintf(result_file, "input[%llu] = %.6f + %.6fi\n", 
                   i, buffer_input[2*i], buffer_input[2*i+1]);
        }
        fprintf(result_file, "\n");
    }

    printf("Input data (N=%llu):\n", N);
    for (uint64_t i = 0; i < N; i++) {
        printf("input[%llu] = %.6f + %.6fi\n", 
               i, buffer_input[2*i], buffer_input[2*i+1]);
    }
    printf("\n");

    // Configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    
    //Setting up FFT configuration
    configuration.FFTdim = 1;
    configuration.size[0] = N;
    configuration.numberBatches = numberBatches;

#if(VKFFT_BACKEND!=5)
    configuration.saveApplicationToString = 0;
#endif

#if(VKFFT_BACKEND==5)
    configuration.device = vkGPU->device;
#else
    configuration.device = &vkGPU->device;
#endif

#if(VKFFT_BACKEND==0)
    configuration.queue = &vkGPU->queue;
    configuration.fence = &vkGPU->fence;
    configuration.commandPool = &vkGPU->commandPool;
    configuration.physicalDevice = &vkGPU->physicalDevice;
    configuration.isCompilerInitialized = isCompilerInitialized;
#elif(VKFFT_BACKEND==3)
    configuration.context = &vkGPU->context;
#elif(VKFFT_BACKEND==4)
    configuration.context = &vkGPU->context;
    configuration.commandQueue = &vkGPU->commandQueue;
    configuration.commandQueueID = vkGPU->commandQueueID;
#elif(VKFFT_BACKEND==5)
    configuration.queue = vkGPU->queue;
#endif

    // Allocate buffer for the input/output data
    uint64_t bufferSize = sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;
    
#if(VKFFT_BACKEND==0)
    VkBuffer buffer = {};
    VkDeviceMemory bufferDeviceMemory = {};
    resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, 
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                           VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
    cuFloatComplex* buffer = 0;
    res = cudaMalloc((void**)&buffer, bufferSize);
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
    hipFloatComplex* buffer = 0;
    res = hipMalloc((void**)&buffer, bufferSize);
    if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
    cl_mem buffer = 0;
    buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
    void* buffer = 0;
    ze_device_mem_alloc_desc_t device_desc = {};
    device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==5)
    MTL::Buffer* buffer = 0;
    buffer = vkGPU->device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
    configuration.buffer = &buffer;
#endif

    configuration.bufferSize = &bufferSize;

    // Initialize VkFFT application
    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    // Transfer input data to GPU
#if(VKFFT_BACKEND==0)
    resFFT = transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
    res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
    res = hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
    if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, buffer, CL_TRUE, 0, bufferSize, buffer_input, 0, NULL, NULL);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
    res = zeCommandListAppendMemoryCopy(vkGPU->commandList, buffer, buffer_input, bufferSize, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==5)
    // Metal implementation would need command buffer setup
#endif

    // Perform forward FFT
    VkFFTLaunchParams launchParams = {};
#if(VKFFT_BACKEND==0)
    launchParams.commandBuffer = &vkGPU->commandBuffer;
#elif(VKFFT_BACKEND==1)
    // CUDA doesn't need special launch params
#elif(VKFFT_BACKEND==2)
    // HIP doesn't need special launch params
#elif(VKFFT_BACKEND==3)
    launchParams.commandQueue = &vkGPU->commandQueue;
#elif(VKFFT_BACKEND==4)
    launchParams.commandList = &vkGPU->commandList;
#elif(VKFFT_BACKEND==5)
    // Metal would need command buffer
#endif

    resFFT = VkFFTAppend(&app, -1, &launchParams);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
    resFFT = submitCommandBuffer(vkGPU);
#elif(VKFFT_BACKEND==3)
    res = clFinish(vkGPU->commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#elif(VKFFT_BACKEND==4)
    res = zeCommandQueueExecuteCommandLists(vkGPU->commandQueue, 1, &vkGPU->commandList, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT64_MAX);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif

    // Transfer result back to CPU
    float* buffer_output = (float*)malloc(bufferSize);
    if (!buffer_output) return VKFFT_ERROR_MALLOC_FAILED;

#if(VKFFT_BACKEND==0)
    resFFT = transferDataToCPU(vkGPU, buffer_output, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
    res = cudaMemcpy(buffer_output, buffer, bufferSize, cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
    res = hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
    if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
    res = clEnqueueReadBuffer(vkGPU->commandQueue, buffer, CL_TRUE, 0, bufferSize, buffer_output, 0, NULL, NULL);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
    res = zeCommandListAppendMemoryCopy(vkGPU->commandList, buffer_output, buffer, bufferSize, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==5)
    // Metal implementation
#endif

    // Print FFT output results
    if (result_file) {
        fprintf(result_file, "FFT output results:\n");
        for (uint64_t i = 0; i < N; i++) {
            fprintf(result_file, "output[%llu] = %.6f + %.6fi\n", 
                   i, buffer_output[2*i], buffer_output[2*i+1]);
        }
        fprintf(result_file, "\n");
        
        // Calculate magnitudes
        fprintf(result_file, "FFT output magnitudes:\n");
        for (uint64_t i = 0; i < N; i++) {
            float magnitude = sqrt(buffer_output[2*i] * buffer_output[2*i] + 
                                 buffer_output[2*i+1] * buffer_output[2*i+1]);
            fprintf(result_file, "magnitude[%llu] = %.6f\n", i, magnitude);
        }
        fprintf(result_file, "\n");
    }

    printf("FFT output results:\n");
    for (uint64_t i = 0; i < N; i++) {
        printf("output[%llu] = %.6f + %.6fi\n", 
               i, buffer_output[2*i], buffer_output[2*i+1]);
    }
    printf("\n");

    printf("FFT output magnitudes:\n");
    for (uint64_t i = 0; i < N; i++) {
        float magnitude = sqrt(buffer_output[2*i] * buffer_output[2*i] + 
                              buffer_output[2*i+1] * buffer_output[2*i+1]);
        printf("magnitude[%llu] = %.6f\n", i, magnitude);
    }

    // Perform inverse FFT to verify
    VkFFTLaunchParams ilaunchParams = {};
#if(VKFFT_BACKEND==0)
    ilaunchParams.commandBuffer = &vkGPU->commandBuffer;
#elif(VKFFT_BACKEND==3)
    ilaunchParams.commandQueue = &vkGPU->commandQueue;
#elif(VKFFT_BACKEND==4)
    ilaunchParams.commandList = &vkGPU->commandList;
#endif

    resFFT = VkFFTAppend(&app, 1, &ilaunchParams);  // inverse FFT
    if (resFFT != VKFFT_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
    resFFT = submitCommandBuffer(vkGPU);
#elif(VKFFT_BACKEND==3)
    res = clFinish(vkGPU->commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#elif(VKFFT_BACKEND==4)
    res = zeCommandQueueExecuteCommandLists(vkGPU->commandQueue, 1, &vkGPU->commandList, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    res = zeCommandQueueSynchronize(vkGPU->commandQueue, UINT64_MAX);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif

    // Transfer inverse FFT result back
    float* buffer_inverse = (float*)malloc(bufferSize);
    if (!buffer_inverse) return VKFFT_ERROR_MALLOC_FAILED;

#if(VKFFT_BACKEND==0)
    resFFT = transferDataToCPU(vkGPU, buffer_inverse, &buffer, bufferSize);
#elif(VKFFT_BACKEND==1)
    res = cudaMemcpy(buffer_inverse, buffer, bufferSize, cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
    res = hipMemcpy(buffer_inverse, buffer, bufferSize, hipMemcpyDeviceToHost);
    if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
    res = clEnqueueReadBuffer(vkGPU->commandQueue, buffer, CL_TRUE, 0, bufferSize, buffer_inverse, 0, NULL, NULL);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==4)
    res = zeCommandListAppendMemoryCopy(vkGPU->commandList, buffer_inverse, buffer, bufferSize, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;
#endif

    // Normalize inverse FFT results (divide by N)
    for (uint64_t i = 0; i < N; i++) {
        buffer_inverse[2*i] = buffer_inverse[2*i] / (float)N;       // Real part
        buffer_inverse[2*i+1] = buffer_inverse[2*i+1] / (float)N;   // Imaginary part
    }

    // Print inverse FFT results and calculate error
    if (result_file) {
        fprintf(result_file, "Inverse FFT results (should match input):\n");
        fprintf(result_file, "Comparison with original input:\n");
        float max_error = 0.0f;
        for (uint64_t i = 0; i < N; i++) {
            float real_error = fabs(buffer_inverse[2*i] - buffer_input[2*i]);
            float imag_error = fabs(buffer_inverse[2*i+1] - buffer_input[2*i+1]);
            if (real_error > max_error) max_error = real_error;
            if (imag_error > max_error) max_error = imag_error;
            
            fprintf(result_file, "inverse[%llu] = %.6f + %.6fi (original: %.6f + %.6fi, error: %.2e + %.2ei)\n", 
                   i, buffer_inverse[2*i], buffer_inverse[2*i+1], 
                   buffer_input[2*i], buffer_input[2*i+1],
                   real_error, imag_error);
        }
        fprintf(result_file, "\nMaximum error: %.2e\n", max_error);
        if (max_error < 1e-5) {
            fprintf(result_file, "VERIFICATION PASSED: Error is within acceptable range\n");
        } else {
            fprintf(result_file, "VERIFICATION FAILED: Error is too large\n");
        }
    }

    printf("\nInverse FFT results (should match input):\n");
    float max_error = 0.0f;
    for (uint64_t i = 0; i < N; i++) {
        float real_error = fabs(buffer_inverse[2*i] - buffer_input[2*i]);
        float imag_error = fabs(buffer_inverse[2*i+1] - buffer_input[2*i+1]);
        if (real_error > max_error) max_error = real_error;
        if (imag_error > max_error) max_error = imag_error;
        
        printf("inverse[%llu] = %.6f + %.6fi (original: %.6f + %.6fi, error: %.2e + %.2ei)\n", 
               i, buffer_inverse[2*i], buffer_inverse[2*i+1], 
               buffer_input[2*i], buffer_input[2*i+1],
               real_error, imag_error);
    }
    printf("\nMaximum error: %.2e\n", max_error);
    if (max_error < 1e-5) {
        printf("VERIFICATION PASSED: Error is within acceptable range\n");
    } else {
        printf("VERIFICATION FAILED: Error is too large\n");
    }

    // Cleanup
    if (result_file) {
        fclose(result_file);
#if (VKFFT_BACKEND==1)
        printf("\nResults saved to fft_verification_results_cuda.txt\n");
#elif (VKFFT_BACKEND==3)
        printf("\nResults saved to fft_verification_results_opencl.txt\n");
#else
        printf("\nResults saved to fft_verification_results.txt\n");
#endif
    }

    deleteVkFFT(&app);
    
#if(VKFFT_BACKEND==0)
    vkDestroyBuffer(vkGPU->device, buffer, NULL);
    vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
    cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
    hipFree(buffer);
#elif(VKFFT_BACKEND==3)
    clReleaseMemObject(buffer);
#elif(VKFFT_BACKEND==4)
    zeMemFree(vkGPU->context, buffer);
#elif(VKFFT_BACKEND==5)
    buffer->release();
#endif

    free(buffer_input);
    free(buffer_output);
    free(buffer_inverse);
    
    return resFFT;
}
