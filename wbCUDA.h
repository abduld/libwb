
#ifndef __WB_CUDA_H__
#define __WB_CUDA_H__

#include "cuda.h"
#include "cuda_runtime.h"

extern size_t _cudaMallocSize;

cudaError_t wbCUDAMalloc(void ** devPtr, size_t sz);
cudaError_t wbCUDAFree(void * mem);

#endif /* __WB_CUDA_H__ */

