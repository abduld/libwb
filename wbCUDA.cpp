
#include "wb.h"
#ifdef WB_USE_CUDA


size_t _cudaMallocSize = 0;

wbCUDAMemory_t _cudaMemoryList[1024];
int _cudaMemoryListIdx = 0;

#endif /* WB_USE_CUDA */
