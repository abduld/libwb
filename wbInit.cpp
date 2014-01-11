
#include    <wb.h>
#include    <wbCUDA.h>

#ifndef WB_DEFAULT_HEAP_SIZE
#define WB_DEFAULT_HEAP_SIZE (1024*1024*120)
#endif /* WB_DEFAULT_HEAP_SIZE */


static bool _initializedQ = wbFalse;


#ifndef _MSC_VER
__attribute__ ((__constructor__))
#endif /* _MSC_VER */
void wb_init(void) {
    if (_initializedQ == wbTrue) {
        return ;
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

#ifdef _MSC_VER
    QueryPerformanceFrequency((LARGE_INTEGER*) &_hrtime_frequency);
#endif

    _hrtime();

    _timer = wbTimer_new();
    _logger = wbLogger_new();
    _initializedQ = wbTrue;

    wbFile_init();

    solutionJSON = NULL;

    atexit(wb_atExit);
}


