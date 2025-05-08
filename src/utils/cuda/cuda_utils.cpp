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

#include <cuda_utils.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#endif


// TODO: remove:
// #define HAVE_CUDA 1
// #define  HAVE_CUMEMRETAINALLOCATIONHANDLE 1
// #define HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST 1


/****************************************
 * CUDA nixlCudaPtr class
*****************************************/

#define HAVE_CUDA

#ifdef HAVE_CUDA

class nixlCudaMemCtxImpl : public nixlCudaMemCtx {
private:
    CUcontext ctx;
    bool ctxWasSet;

    nixl_status_t isVmm(void *address);
    nixl_status_t isCuda(void *address);
    nixl_status_t unsetMemCtx() override;

public:

    nixlCudaMemCtxImpl() : nixlCudaMemCtx()
    {    }

    ~nixlCudaPtrImpl() override {
        if (MEM_VMM_DEV == memType) {
            nixl_status_t status = releaseVmmCudaCtx(_devId);
            if (NIXL_SUCCESS != status) {
                NIXL_ERROR << "Failed to release CUDA context";
            }
        }
    }

    nixl_status_t enableAddr(const void *address) override;

    nixl_status_t set() override;
};

#endif

/****************************************
 * Static nixlCudaPtr functions
*****************************************/

#ifdef HAVE_CUDA

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaPtrImpl
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT true

#else

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaPtrCtx
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT false

#endif

bool nixlCudaPtrCtx::vramIsSupported()
{
    return NIXL_CUDA_PTR_CTX_VRAM_SUPPORT;
}

std::unique_ptr<nixlCudaMemCtx>
nixlCudaMemCtx::nixlCudaMemCtxInit()
{
    // Environment fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        // If the workarounf is disabled - return the dummy class
        NIXL_INFO << "WARNING: disabling CUDA address workaround";
        return std::make_unique<nixlCudaMemCtx>();
    } else {
        return std::make_unique<NIXL_CUDA_PTR_CTX_CLASS>();
    }
}


/****************************************
 * CUDA nixlCudaPtr class implementation
*****************************************/

#ifdef HAVE_CUDA

nixl_status_t
nixlCudaPtrImpl::queryVmm(void *address, memory_t &type, uint64_t &id)
{
    nixl_status_t ret = NIXL_ERR_NOT_FOUND;

#if HAVE_CUMEMRETAINALLOCATIONHANDLE
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;
    CUresult result;

    /* Check if memory is allocated using VMM API and see if host memory needs
     * to be treated as pinned device memory */
    result = cuMemRetainAllocationHandle(&alloc_handle, (void*)address);
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

    id = (CUdevice)prop.location.id;
    switch (prop.location.type) {
    case CU_MEM_LOCATION_TYPE_DEVICE:
        type = MEM_VMM_DEV;
        break;
#if HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST
    case CU_MEM_LOCATION_TYPE_HOST:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
        // Currently not supported
        //type = MEM_VMM_HOST;
#endif
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
retainVmmCudaCtx(uint64_t id, CUcontext &newCtx) const
{
    unsigned int flags;
    int active;

    CUresult result = cuDevicePrimaryCtxGetState(id, &flags, &active);
    if (result != CUDA_SUCCESS) {
        // TODO: log error
        return NIXL_ERR_UNKNOWN;
    }

    if (!active) {
        // TODO: Not supported at the moment. In most cases it is set
        // FIXME: Allocate a new context?
        return NIXL_ERR_INVALID_PARAM;
    }

    result = cuDevicePrimaryCtxRetain(&newCtx, id);
    if (result != CUDA_SUCCESS) {
        // TODO: log error
        return NIXL_ERR_UNKNOWN;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
releaseVmmCudaCtx(uint64_t id) const
{
    CUresult result = cuDevicePrimaryCtxRelease(id);
    return (CUDA_SUCCESS == result) ? NIXL_SUCCESS : NIXL_ERR_UNKNOWN;
}

nixl_status_t
nixlCudaPtrImpl::queryCuda(void *address, memory_t &type, uint64_t &id, CUcontext &newCtx)
{
    CUmemorytype cudaMemType = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cudaMemType;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &id;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &newCtx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);
    if (CUDA_SUCCESS != result) {
        return NIXL_ERR_NOT_FOUND;
    }

    switch(cudaMemType) {
    case CU_MEMORYTYPE_DEVICE:
        type = MEM_DEV;
        break;
    case CU_MEMORYTYPE_HOST:
        type = MEM_HOST;
    case CU_MEMORYTYPE_ARRAY:
        // TODO: how should this case be processed?
        return NIXL_ERR_INVALID_PARAM;
    default:
        return NIXL_ERR_NOT_FOUND;
    }

    // TODO: what to do if the memory "is_managed"?

    return NIXL_SUCCESS;
}

nixl_status_t 
nixlCudaPtrImpl::enableAddr(const void *address)
{
    nixl_status_t status;
    memory_t addrMemType = MEM_NONE;
    memory_t tmpMemType;
    CUcontext newCtx;
    uint64_t newDevId;

    status = checkVmm(address, tmpMemType, newDevId);
    if (NIXL_SUCCESS == status) {
        addrMemType = tmpMemType;
    } else if (status == NIXL_ERR_NOT_FOUND) {
        status = checkCuda(address, tmpMemType, newDevId, newCtx);
        if (NIXL_SUCCESS == status) {
            addrMemType = tmpMemType;
        }
    }

    if (status != NIXL_SUCCESS) {
        return status;
    }

    if (MEM_NONE == memType) {
        // Initialize the context
        switch(addrMemType) {
        case MEM_HOST:
            break;
        case MEM_VMM_DEV:
            status = retainVmmCudaCtx(newDevId, newCtx);
            if (NIXL_SUCCESS != status) {
                return NIXL_ERR_UNKNOWN;
            }
            /* fall through */
        case MEM_DEV:
            ctx = newCtx;
            _devId = newDevId;
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }
        // All set successfully =>  safe to set memType
        memType = addrMemType;
        return NIXL_IN_PROG;
    } else {
        // Must match existing present
        if (memType != addrMemType) {
            return NIXL_ERR_INVALID_PARAM;
        }

        if (newDevId != _devId) {
            return NIXL_ERR_INVALID_PARAM;
        }

        if (MEM_VMM_DEV == addrMemType) {
            status = retainVmmCudaCtx(newDevId, newCtx);
            if (NIXL_SUCCESS != status) {
                return NIXL_ERR_UNKNOWN;
            }
        }
        // Status shoud be NIXL_SUCCESS
        if (ctx != newCtx) {
            status = NIXL_ERR_INVALID_PARAM;
        }
        if (MEM_VMM_DEV == addrMemType) {
            releaseVmmCudaCtx(newDevId);
        }
        return status;
    }
}

nixl_status_t
nixlCudaPtrImpl::set()
{
    CUresult result;

    switch (mem_type) {
    case MEM_HOST:
        return NIXL_SUCCESS;
    case MEM_DEV:
    case MEM_VMM_DEV: {
        result = cuCtxSetCurrent(ctx);
        if (CUDA_SUCCESS != result) {
            // TODO: something like NIXL_ERR_CMD_FAILED
            // would be more appropriate
            return NIXL_ERR_NOT_SUPPORTED;
        }
        return NIXL_SUCCESS;
    }
    case MEM_VMM_HOST:
        // TODO: Not supported at the moment
        // fall through
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

#endif
