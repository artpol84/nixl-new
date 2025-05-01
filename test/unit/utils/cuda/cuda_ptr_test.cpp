/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cuda/cuda_utils.h>

using namespace std;

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;

    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

#endif

static int allocateCUDA(int dev_id, size_t len, void* &addr)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;

    checkCudaError(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer 0");
    cudaQueryAddr(addr, is_dev, dev, ctx);
    std::cout << "CUDA addr: " << std::hex << addr << " dev=" << std::dec << dev
        << " ctx=" << std::hex << ctx << std::dec << std::endl;
    return 0;
}

void releaseCUDA(int dev_id, void* addr)
{
    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaFree(addr), "Failed to allocate CUDA buffer 0");
}

#ifdef HAVE_CUDA_VMM

#define ROUND_UP(value, granularity) ((((value) + (granularity) - 1) / (granularity)) * (granularity))
static size_t __attribute__((unused)) padded_size = 0;
static CUmemGenericAllocationHandle __attribute__((unused)) handle;

static int allocateVMM(int dev_id, size_t len, void* &addr, )
{
    CUdeviceptr addr = 0;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};

    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = dev_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;

    // Get the allocation granularity
    checkCudaError(cuMemGetAllocationGranularity(&granularity, prop,
                                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "Failed to get allocation granularity");
    std::cout << "Granularity: " << granularity << std::endl;

    padded_size = ROUND_UP(len, granularity);
    checkCudaError(cuMemCreate(&handle, padded_size, &prop, 0),
                         "Failed to create allocation");

    // Reserve the memory address
    checkCudaError(cuMemAddressReserve(&addr, padded_size,
                                        granularity, 0, 0),
                   "Failed to reserve address");

    // Map the memory
    checkCudaError(cuMemMap(addr, padded_size, 0, handle, 0),
                   "Failed to map memory");

    std::cout << "Address: " << std::hex << std::showbase << addr
              << " Buffer size: " << std::dec << buffer_size
              << " Padded size: " << std::dec << padded_size << std::endl;

    // Set the memory access rights
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaError(cuMemSetAccess(addr, buffer_size, &access, 1),
                   "Failed to set access");
}

void releaseVMM(int dev_id, size_t len, void* addr)
{
    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cuMemUnmap(addr, len),
                  "Failed to unmap memory");
    checkCudaError(cuMemRelease(handle),
                   "Failed to release memory");
    checkCudaError(cuMemAddressFree(addr, padded_size),
                   "Failed to free reserved address");
}

#endif


int main()
{
    void *address;
    size_t len = 1024;

    /* Discover environemnt */
    int ngpus;
    cudaGetDeviceCount(&ngpus);

    if (!ngpus) {
        cout << "No GPGPU devices detected, nothing to test!" << endl;
        return 0;
    }

    /* Test regular CUDA malloc */
    {
        address = malloc(len);
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType == nixlCudaPtrCtx::MEM_HOST);
        free(address);
    }

    /* Test regular CUDA malloc */
    {
        assert(0 == allocateCUDA(0, len, addres));
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType == nixlCudaPtrCtx::MEM_DEV);
        assert(0 == releaseCUDA(0, address));
    }

if HAVE_CUDA_VMM
    /* Test regular CUDA malloc */
    {
        assert(0 == allocateVMM(0, len, addres));
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType == nixlCudaPtrCtx::MEM_VMM_DEV);1
        releaseVMM(0, len, address);
    }
#endif

}
