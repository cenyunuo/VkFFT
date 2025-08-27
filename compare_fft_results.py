#!/usr/bin/env python3
"""
Build VkFFT with CUDA and OpenCL backends, run verification test,
and compare FFT outputs to ensure they are numerically similar.
"""
import subprocess
import re
import math
import os


def run(cmd, **kwargs):
    print('Running', ' '.join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def build_and_run(backend, build_dir):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    run(["cmake", "-S", ".", "-B", build_dir, f"-DVKFFT_BACKEND={backend}"])
    run(["cmake", "--build", build_dir, "-j"])
    exe = os.path.join(build_dir, "VkFFT_TestSuite")
    run([exe, "-vkfft", "9999"])
    if backend == 1:
        return "fft_verification_results_cuda.txt"
    elif backend == 3:
        return "fft_verification_results_opencl.txt"
    else:
        raise ValueError("Unsupported backend")


def parse_results(path):
    values = []
    with open(path) as f:
        for line in f:
            if line.startswith("output["):
                m = re.search(r"output\\[\\d+\\] = ([^ ]+) \+ ([^ ]+)i", line)
                if m:
                    values.append(complex(float(m.group(1)), float(m.group(2))))
    return values


cuda_file = build_and_run(1, "build")
opencl_file = build_and_run(3, "build_opencl")

cuda_vals = parse_results(cuda_file)
opencl_vals = parse_results(opencl_file)

if len(cuda_vals) != len(opencl_vals):
    raise RuntimeError("Result lengths differ")

max_diff = max(abs(a - b) for a, b in zip(cuda_vals, opencl_vals))
print("Max difference between CUDA and OpenCL FFT results:", max_diff)

assert max_diff < 1e-4, "FFT results differ too much: %g" % max_diff
print("CUDA and OpenCL FFT outputs are sufficiently close.")
