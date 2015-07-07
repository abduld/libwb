
#include "wb.h"
#ifdef WB_USE_CUDA

int _cudaMemoryListIdx = 0;

size_t _cudaMallocSize = 0;

wbCUDAMemory_t _cudaMemoryList[_cudaMemoryListSize];

char *wbRandom_list(size_t sz) {
  size_t ii;
  char *rands = wbNewArray(char, sz);
  int *irands = (int *)rands;
  for (ii = 0; ii < sz / sizeof(int); ii++) {
    irands[ii] = rand();
  }
  while (ii < sz) {
    rands[ii] = (char)(rand() % 255);
    ii++;
  }
  return rands;
}

#endif /* WB_USE_CUDA */
