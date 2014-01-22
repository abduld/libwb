
#include <wb.h>
#include <wbCUDA.h>

#ifndef WB_DEFAULT_HEAP_SIZE
#define MB (1 << 20)
const size_t WB_DEFAULT_HEAP_SIZE = (50 * MB);
#undef MB
#endif /* WB_DEFAULT_HEAP_SIZE */

static bool _initializedQ = wbFalse;

#ifndef _MSC_VER
__attribute__((__constructor__))
#endif /* _MSC_VER */
    void wb_init(void) {
  if (_initializedQ == wbTrue) {
    return;
  }

#ifdef WB_USE_CUDA
  cuInit(0);

  /* Select a random GPU */

  {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    srand(time(NULL));
    cudaSetDevice(rand() % deviceCount);
  }

#endif /* WB_USE_CUDA */

#ifdef WB_USE_CUSTOM_MALLOC
  wbMemoryManager_new(WB_DEFAULT_HEAP_SIZE);
#endif /* WB_USE_CUSTOM_MALLOC */

#ifdef _MSC_VER
  QueryPerformanceFrequency((LARGE_INTEGER *)&_hrtime_frequency);
#endif /* _MSC_VER */

  _hrtime();

  _timer = wbTimer_new();
  _logger = wbLogger_new();
  _initializedQ = wbTrue;

  wbFile_init();

#ifdef WB_USE_SANDBOX
  wbSandbox_new();
#endif /* WB_USE_SANDBOX */

  solutionJSON = NULL;

  atexit(wb_atExit);
}
