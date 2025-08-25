# VkFFT Verification System

This repository contains an enhanced VkFFT test suite with detailed FFT result verification capabilities. The verification system allows you to validate the correctness of FFT computations across different backends (CUDA, OpenCL, etc.) and examine detailed numerical results.

## Features

- **Comprehensive FFT Verification**: Test forward and inverse FFT operations with detailed error analysis
- **Multiple Backend Support**: Compare results between CUDA and OpenCL implementations
- **Detailed Output**: View input data, FFT outputs, magnitudes, and inverse FFT results
- **Error Analysis**: Automatic calculation of numerical errors and pass/fail determination
- **File Output**: Results are saved to text files for further analysis
- **Integration**: Seamlessly integrated into the existing VkFFT test suite

## Quick Start

### Prerequisites

- CUDA Toolkit (for CUDA backend)
- OpenCL SDK (for OpenCL backend)
- CMake 3.10 or higher
- C++ compiler with C++11 support

### Building

1. **Build CUDA Version:**
```bash
mkdir build
cd build
cmake -DVKFFT_BACKEND=1 ..
make -j$(nproc)
```

2. **Build OpenCL Version:**
```bash
mkdir build_opencl
cd build_opencl
cmake -DVKFFT_BACKEND=3 ..
make -j$(nproc)
```

### Running Verification Tests

Execute the verification test using test ID `9999`:

```bash
# Run CUDA verification
./build/VkFFT_TestSuite -vkfft 9999

# Run OpenCL verification (if OpenCL runtime is available)
./build_opencl/VkFFT_TestSuite -vkfft 9999
```

## Verification Test Details

### Test Configuration

- **FFT Size**: N = 16 (configurable in source code)
- **Data Type**: Single precision complex numbers (float)
- **Input Pattern**: Linear sequence [0, 1, 2, ..., 15] + 0i
- **Operations**: Forward FFT → Inverse FFT → Error calculation

### Output Information

The verification test provides:

1. **Input Data**: Original complex input values
2. **FFT Output**: Forward FFT transformation results (complex values)
3. **Magnitudes**: Magnitude spectrum of FFT output
4. **Inverse FFT**: Inverse transformation results
5. **Error Analysis**: Element-wise error comparison with original input
6. **Verification Status**: PASS/FAIL based on maximum error threshold

### Sample Output

```
VkFFT Verification Test - Simple FFT with result output
Input data (N=16):
input[0] = 0.000000 + 0.000000i
input[1] = 1.000000 + 0.000000i
...

FFT output results:
output[0] = 120.000000 + 0.000000i
output[1] = -8.000002 + 40.218712i
...

Inverse FFT results (should match input):
inverse[0] = 0.000000 + 0.000000i (original: 0.000000 + 0.000000i, error: 0.00e+00 + 2.38e-07i)
...

Maximum error: 9.54e-07
VERIFICATION PASSED: Error is within acceptable range
```

## File Structure

```
VkFFT/
├── sample_verification.h          # Verification module header
├── sample_verification.cpp        # Verification implementation
├── VkFFT_TestSuite.cpp           # Main test suite (modified)
├── CMakeLists.txt                 # Build configuration (updated)
├── build/                         # CUDA build directory
├── build_opencl/                  # OpenCL build directory
├── fft_verification_results_cuda.txt  # CUDA verification results
└── README_VERIFICATION.md        # This file
```

## Implementation Details

### Verification Algorithm

1. **Initialization**: Create input data with known pattern
2. **Memory Management**: Allocate GPU memory for input/output buffers
3. **Forward FFT**: Execute FFT transformation
4. **Result Capture**: Copy FFT results back to CPU
5. **Inverse FFT**: Execute inverse FFT transformation
6. **Normalization**: Apply 1/N scaling for proper inverse FFT
7. **Error Calculation**: Compare inverse results with original input
8. **Reporting**: Generate detailed output and save to file

### Error Threshold

- **Pass Threshold**: Maximum error < 1e-5
- **Typical Results**: ~1e-6 to 1e-7 for single precision
- **Error Sources**: Floating-point arithmetic, GPU precision limits

### Backend-Specific Features

#### CUDA Backend (VKFFT_BACKEND=1)
- Uses CUDA memory management (cudaMalloc, cudaMemcpy)
- Supports all modern NVIDIA GPUs
- Optimized for CUDA compute capabilities

#### OpenCL Backend (VKFFT_BACKEND=3)
- Uses OpenCL buffer management (clCreateBuffer, clEnqueueReadBuffer)
- Cross-platform compatibility (NVIDIA, AMD, Intel)
- Requires proper OpenCL runtime installation

## Customization

### Changing FFT Size

Edit `sample_verification.cpp` line ~45:
```cpp
uint64_t N = 64;  // Change from 16 to desired size
```

### Modifying Input Data

Edit the input generation loop in `sample_verification.cpp`:
```cpp
for (uint64_t i = 0; i < N; i++) {
    buffer_input[2*i] = (float)sin(2.0 * M_PI * i / N);  // Custom pattern
    buffer_input[2*i+1] = 0.0f;
}
```

### Adjusting Error Threshold

Modify the threshold in the verification logic:
```cpp
if (max_error < 1e-6) {  // More strict threshold
    printf("VERIFICATION PASSED\n");
}
```

## Troubleshooting

### Common Issues

1. **OpenCL Platform Not Found**
   ```
   Number of platforms: 0
   ```
   - Install OpenCL runtime for your GPU
   - For NVIDIA: Install CUDA Toolkit
   - For AMD: Install ROCm or AMD OpenCL SDK

2. **Compilation Errors**
   - Ensure CUDA/OpenCL headers are in system path
   - Check CMake backend selection (-DVKFFT_BACKEND=1 or 3)

3. **Memory Allocation Failures**
   - Reduce FFT size for systems with limited GPU memory
   - Check available GPU memory using nvidia-smi

### Performance Notes

- CUDA backend typically offers better performance on NVIDIA hardware
- OpenCL backend provides broader compatibility but may have slightly lower performance
- Large FFT sizes (N > 1024) may require significant GPU memory

## Contributing

To add new verification tests:

1. Create new functions in `sample_verification.cpp`
2. Add test cases to the switch statement in `VkFFT_TestSuite.cpp`
3. Update help text and documentation
4. Test with both CUDA and OpenCL backends

## License

This verification system follows the same MIT license as the main VkFFT library.

## Acknowledgments

- Built upon the excellent VkFFT library by Dmitrii Tolmachev
- Supports CUDA, OpenCL, Vulkan, and other compute backends
- Designed for high-performance FFT verification and testing
