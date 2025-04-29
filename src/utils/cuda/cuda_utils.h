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

/* forward declaration for the internal CUDA data structure */


enum class nixl_cuda_mem_t {
    NIXL_CUDA_MEM_NONE,
    NIXL_CUDA_MEM_HOST,
    NIXL_CUDA_MEM_DEV,
    NIXL_CUDA_MEM_VMM_HOST,
    NIXL_CUDA_MEM_VMM_DEV,
    NIXL_CUDA_MEM_INVALID,
};

/****************************************
 * Pointer Context
*****************************************/

class nixlCudaPtrCtx {
protected:
    void *address;
    nixl_cuda_mem_t mem_type;
    int devId;

    /* To be used in derived classes */
    inline virtual bool
    internalCmp(const nixlCudaPtrCtx &lhs, 
                const nixlCudaPtrCtx &rhs) {
        return true;
    }

public:
    nixlCudaPtrCtx(void *addr) :
                   address(addr), type(NIXL_CUDA_PTR_HOST), 
                   supportVram(false), devId(0)
    { /* Empty body */ }

    virtual ~nixlCudaPtrCtx() = default;

    inline nixl_cuda_mem_t getMemType() {
        return mem_type;
    }

    inline int getDevId() {
        return devId;
    }

    virtual void setMemCtx() {
        // no-op for non-CUDA case
    }

    virtual void unsetMemCtx() {
        // no-op for non-CUDA case
    }

    inline bool operator==(const nixlCudaPtrCtx &lhs, 
                           const nixlCudaPtrCtx &rhs) {
        return lhs.type == rhs.type && internalCmp(lhs, rhs);
    }


    static bool vramIsSupported();
    static std::unique_ptr<nixlCudaPtrCtx> *nixlCudaPtrInit(void *address);

};

