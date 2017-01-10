#! /bin/bash
nvcc -ptx cuLaunchKernel.cu -o cuLaunchKernel.ptx
nvcc -cubin cuLaunchKernel.cu -o cuLaunchKernel.cubin
cuobjdump -elf cuLaunchKernel.cubin
nvcc cuLaunchKernel.cpp -o cuLaunchKernel.exe -lcuda --cudart=shared
