#include<cstdio>

extern "C" {
    __global__ void func_0() {
        int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __global__ void func_1(int a) {
        int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __global__ void func_1bis(int a) {
        int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __global__ void func_2(int a, int b) {
        int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    }
    __global__ void func_3(int a, int b, int *c) {
        int tId = (blockIdx.x * blockDim.x) + threadIdx.x;
    }
}
