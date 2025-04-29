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

#include "cuda_utils.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cudart.h>

#endif

// TODO: remove:
#define HAVE_CUDA 1
#define  HAVE_CUMEMRETAINALLOCATIONHANDLE 1
#define HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST 1

/****************************************
 * CUDA nixlCudaPtr version
*****************************************/

#ifdef HAVE_CUDA

class nixlCudaPtrImpl : public nixlCudaPtr {
private:
    CUcontext ctx;

    nixl_status_t initVmm(void *address);
    nixl_status_t initCuda(void *address);


    /* To be used in derived classes */
    inline bool
    intCompare(const nixlPtrCtxBase &_lhs,
               const nixlPtrCtxBase &_rhs) override {
        const nixlPtrCtxCuda &lhs = *(const nixlPtrCtxCuda *)_lhs;
        const nixlPtrCtxCuda &rhs = *(const nixlPtrCtxCuda *)_rhs;
        switch (mem_type) {
        case NIXL_CUDA_PTR_HOST:
            return true;
        case NIXL_CUDA_PTR_DEV:
            return (lhs.dev == rhs.dev) &&
                   (lhs.ctx == rhs.ctx);
            break;
        case NIXL_CUDA_PTR_VMM_DEV:
        case NIXL_CUDA_PTR_VMM_HOST:
            // TODO: check what is required 
            return (lhs.dev == rhs.dev);
            break;
        default:
            // TODO error log
            return false;
        }
    }

public:

    nixlCudaPtrCtx(void *addr) : nixlCudaPtrCtx (addr)
    {
        nixl_status_t status;

        supportVram = true;
    
        // Test VMM allocations first
        status = initVmm(address);
        if (status == NIXL_SUCCESS) {
            // This is VMM allocation, the class is initialized
            return;
        }

        if (status != NIXL_ERR_NOT_SUPPORT) {
            // Unexpected error
            // TODO: throw status;
        }

        // Continue with CUDA and Host allocations
        status = initCuda(address);
        if (status == NIXL_SUCCESS) {
            // Everything is initialized
            return;
        } else {
            // Unexpected error
            // TODO: throw status;
        }
    }

    ~nixlCudaPtrCtx() override {
    }

    void setMemCtx() override;
    void unsetMemCtx() override;

    inline bool vramIsSupported() {
        return true;
    }

    inline nixl_cuda_ptr_t getType() {
        return type;
    }
};

bool nixlPtrCtxBase::vramIsSupported()
{
    return true;
}

std::unique_ptr<nixlCudaPtrCtx> nixlCudaPtrCtxInit(void *address)
{
    std::unique_ptr<nixlCudaPtrCtx> ptr;
    try {
        ptr = std::make_unique<nixlCudaPtrCtxImpl>(address);
    } catch()
    {

    }
    return ptr;
}

#else

#define NIXL_CUDA_PTR_CTX_CLASS nixlPtrCtxBase
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT false

#endif

bool nixlPtrCtxBase::vramIsSupported()
{
    return NIXL_CUDA_PTR_CTX_VRAM_SUPPORT;
}

std::unique_ptr<nixlPtrCtxBase>
nixlPtrCtxBase::nixlCudaPtrCtxInit(void *address)
{
    std::unique_ptr<NIXL_CUDA_PTR_CTX_CLASS> ptr;
    ptr = std::make_unique<NIXL_CUDA_PTR_CTX_CLASS>(address);

    return ptr;
}


#ifdef HAVE_CUDA

nixl_status_t
nixlCudaPtrCtxImpl::checkVmm(void *address)
{
    nixl_status_t ret = NIXL_SUCCESS;

#if HAVE_CUMEMRETAINALLOCATIONHANDLE
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;
    CUresult result;

    /* Check if memory is allocated using VMM API and see if host memory needs
     * to be treated as pinned device memory */
    result = cuMemRetainAllocationHandle(&alloc_handle, (void*)address));
    if (result != CUDA_SUCCESS) {
        return NIXL_ERR_NOT_FOUND;
    }
    // TODO: set the call to cuMemRelease when leaving the scope to avoid GOTO 

    result = cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle);
    if (result != CUDA_SUCCESS) {
        // TODO: log error
        ret = NIXL_ERR_UNKNOWN;
        goto err;
    }

    dev = (CUdevice)prop.location.id;
    switch (prop.location.type) {
#if HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST
    case CU_MEM_LOCATION_TYPE_HOST:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
        /* Do we need to set context in this case? */
        type = NIXL_CUDA_PTR_VMM_HOST;
        break;
#endif
    case CU_MEM_LOCATION_TYPE_DEVICE:
        type = NIXL_CUDA_PTR_VMM_DEV;
        break;
    default:
        // This is VMM memory, but its invalid
        ret = NIXL_ERR_INVALID_PARAM;
        goto err;
    }

err:
    result = cuMemRelease(alloc_handle);
    if (CUDA_SUCCESS != result) {
        // TODO: log error
        if (NIXL_SUCCESS == ret) {
            ret = NIXL_ERR_UNKNOWN;
        }
    }
#endif
    return ret;
}


nixl_status_t
nixlCudaPtrCtxImpl::checkCuda(void *address)
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
    if (CUDA_SUCCESS != result) {
        return NIXL_ERR_INVALID_PARAM;
    }

    switch(mem_type) {
    case CU_MEMORYTYPE_DEVICE:
        type = NIXL_CUDA_PTR_DEV;
        break;
    case CU_MEMORYTYPE_HOST:
        type = NIXL_CUDA_PTR_HOST;
    case CU_MEMORYTYPE_ARRAY:
        // TODO: how should this case be processed?
    default:
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: what to do if the memory "is_managed"?

    return NIXL_SUCCESS;
}

nixl_status_t
nixlCudaPtrCtxImpl::setMemCtx()
{
    CUresult result;

    switch (type) {
    case NIXL_CUDA_PTR_HOST:
        return NIXL_SUCCESS;
    case NIXL_CUDA_PTR_DEV: {
        result = cuCtxSetCurrent(ctx);
        if (CUDA_SUCCESS != result) {
            // TODO: something like NIXL_ERR_CMD_FAILED
            // would be more appropriate
            return NIXL_ERR_NOT_SUPPORTED;
        }
        return NIXL_SUCCESS;
    }
    case NIXL_CUDA_PTR_VMM_DEV: {
        unsigned int flags;
        int active;
    
        result = cuDevicePrimaryCtxGetState(dev, &flags, &active);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }
    
        if (!active) {
            // TODO: Not supported at the moment. In most cases it is set
            // FIXME: Allocate a new context?
            return NIXL_ERR_INVALID_PARAM;
        }
    
        result = cuDevicePrimaryCtxRetain(&ctx, dev);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
    }
    case NIXL_CUDA_PTR_VMM_HOST:
        // TODO: Not supported at the moment
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

nixl_status_t
nixlCudaPtrCtxImpl::unsetMemCtx()
{
    switch (type) {
    case NIXL_CUDA_PTR_HOST:
    case NIXL_CUDA_PTR_DEV:
        return NIXL_SUCCESS;
    case NIXL_CUDA_PTR_VMM_DEV: {
        CUresult result;
        result = cuDevicePrimaryCtxRelease(dev);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
    }
    case NIXL_CUDA_PTR_VMM_HOST:
        // TODO: Not supported at the moment
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

#endif



/****************************************
 * Static nixlCudaPtr functions
*****************************************/

#ifdef HAVE_CUDA

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaPtrImpl
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT true

#else

#define NIXL_CUDA_PTR_CTX_CLASS nixlPtrCtxBase
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT false

#endif

bool nixlCudaPtr::vramIsSupported()
{
    return NIXL_CUDA_PTR_CTX_VRAM_SUPPORT;
}

std::unique_ptr<nixlCudaPtr>
nixlCudaPtr::nixlCudaPtrCtxInit(void *address)
{
    std::unique_ptr<NIXL_CUDA_PTR_CTX_CLASS> ptr;
    ptr = std::make_unique<NIXL_CUDA_PTR_CTX_CLASS>(address);

    return ptr;
}