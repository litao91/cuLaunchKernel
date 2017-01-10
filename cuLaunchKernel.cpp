#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "cuda.h"

using namespace std;


void Dump( const void * mem, unsigned int n ) {
  const char * p = reinterpret_cast< const char *>( mem );
  int j=0;
  int k=0;
  long offset=0;
  std::stringstream sHex,sAscii;
  for ( unsigned int i = 0; i < n; i++ ) {
     int q=int(p[i] & 0xff);
     sHex << setfill('0') << setw(2) << hex << q << " ";
     if (j==7) {
         sHex << " "; 
     }
     if (q>31 && q<127) { 
         sAscii << (char)q ;
     } else {
         sAscii << ".";
     }
     j++;
     if (j==16) {
         unsigned long index=int(offset+(k*16));
         unsigned long addrs=index+(unsigned long)mem;
         std::cout << setfill('0') << setw(8) << hex << addrs <<"|"<< setfill('0') << setw(8) << hex << index << ": " << sHex.str() << "|" << sAscii.str() << "|" << endl;
         sHex.str("");
         sAscii.str("");
         j=0;
         k++;
     }
  }
  std::cout << std::endl;
}

struct gdev_cmem {
                uint64_t addr; 
                uint32_t size; 
                uint32_t offset; 
        };

/* This structure provides a view as an array of long words */
struct myf {
                uint64_t addrs[256];
        };

/* This struct deals with the CUDA kernel */
struct kernel {
    uint32_t v0;
    uint32_t v1;
    uint32_t v2;
    uint64_t v3;
    uint32_t v4;
    uint32_t v5;
    uint32_t v6;
    uint32_t v7;
    uint32_t v8;
    void *module;
    uint32_t size;
    uint32_t v9;
    void *p1;   
};

struct dummy1 {
  void *p0;
  void *p1;
  uint64_t v0;
  uint64_t v1;
  void *p2;
};

/* The function struct!!! */
struct CUfunc_st {
    uint32_t v0;
    uint32_t v1;
    char *name;
    uint32_t v2;
    uint32_t v3;
    uint32_t v4;
    uint32_t v5;
    struct kernel *kernel;
    void *p1;
    void *p2;
    uint32_t v6;
    uint32_t v7;
    uint32_t v8;
    uint32_t v9;
    uint32_t v10;
    uint32_t v11;
    uint32_t v12;
    uint32_t v13;
    uint32_t v14;
    uint32_t v15;
    uint32_t v16;
    uint32_t v17;
    uint32_t v18;
    uint32_t v19;
    uint32_t v20;
    uint32_t v21;
    uint32_t v22;
    uint32_t v23;
    struct dummy1 *p3;
};

int main(){

    // Driver inizialization
    cuInit(0);
    
    // Get the device
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n"); 
        exit(1);
    }

    // Create the CUDA context
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    // Load the module
    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "cuLaunchKernel.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);  
        exit(1); 
    }

    // Get the pointer to CUfunc_st
    CUfunction func;
    res = cuModuleGetFunction(&func, cuModule, "func_1");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

/*
 * This definition is from an old cuda novou driver
struct CUfunc_st {
        struct gdev_kernel kernel;
        struct gdev_cuda_raw_func raw_func;
        struct gdev_list list_entry;
        struct CUmod_st *mod;
};
 */

    // Access to the CUfunc_st as an array of 1024 uint64_t
    struct myf *pmyf=(struct myf *)func;

    // Show on the console the dump of 1024 uint64_t
    cout << "The raw memory pointed by CUfunction" << endl;
    Dump((void *)pmyf,sizeof(struct myf));
    cout << "------------------" << endl;
    
    // Our version of CUfunc_st
    struct CUfunc_st *pFunc=(struct CUfunc_st *)func;

    // Show on the console the decoded CUfunc_st
    cout << "Partally decoded CUfunc_st" << endl;
    cout << "v0:" << pFunc->v0 << endl;
    cout << "v1:" << pFunc->v1 << endl;
    cout << "name:" << pFunc->name << endl;
    cout << "v2:" << pFunc->v2 << endl;
    cout << "v3:" << pFunc->v3 << endl;
    cout << "v4:" << pFunc->v4 << endl;
    cout << "v5:" << pFunc->v5 << endl;

    // Our version of kernel structure
    struct kernel *pKernel=pFunc->kernel;

    cout << "------------------" << endl;
    cout << "pFunc->kernel" << endl;
    Dump((void *)pKernel,sizeof(struct kernel)+256);
    cout << "------------------" << endl;
    cout << "Partially decoded kernel struct" << endl;
    cout << "kernel.v0:" << pKernel->v0 << endl;
    cout << "kernel.v1:" << pKernel->v1 << endl;
    cout << "kernel.v2:" << pKernel->v2 << endl;
    cout << "kernel.v3:" << pKernel->v3 << endl;
    cout << "kernel.v4:" << pKernel->v4 << endl;
    cout << "kernel.v5:" << pKernel->v5 << endl;
    cout << "kernel.v6:" << pKernel->v6 << endl;
    cout << "kernel.v7:" << pKernel->v7 << endl;
    cout << "kernel.v8:" << pKernel->v8 << endl;
    cout << "kernel.size:" << pKernel->size << " size of the module" << endl;
    cout << "kernel.module:" << pKernel->module << " pointer to the module (the ELF binary)" << endl;
    cout << "kernel.v9:" << pKernel->v9 << endl;

    cout << "------------------" << endl;
    cout << "Module dump! Similar to 'hexdump -C cuLaunchKernel.cubin'" << endl;
    Dump((void *)(pKernel->module),pKernel->size);
    cout << "------------------" << endl;
    //cout << "pKernel->p1" << endl;
    //Dump((void *)(pKernel->p1),128);
    //cout << "------------------" << endl;
    //cout << "pFunc->p1" << endl;
    //Dump((void *)(pFunc->p1),128);
    //cout << "------------------" << endl;
    //cout << "pFunc->p2" << endl;
    //Dump((void *)(pFunc->p2),128);
    //cout << "------------------" << endl;
    //cout << "pFunc->p3" << endl;
    //Dump((void *)(pFunc->p3),256);
    //cout << "------------------" << endl;

    //struct dummy1 *pDummy1=(struct dummy1 *)(pFunc->p3);
    //Dump((void *)(pDummy1->p0),128);
    //cout << "------------------" << endl;
    //Dump((void *)(pDummy1->p1),128);
    //cout << "------------------" << endl;
    //Dump((void *)(pDummy1->p2),256);

    int blocks_per_grid = 4;
    int threads_per_block = 5;

    int a=3;
    int b=2;
    int *c;

    void* args[] = { &a, &b,&c };


    res = cuLaunchKernel(func, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

    cuCtxDestroy(cuContext);

    return 0;
}
