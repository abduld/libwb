

#ifndef __WB_H__
#define __WB_H__

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _MSC_VER

// set minimal warning level
#pragma warning(push, 0)
// some warnings still occur at this level
// if necessary, disable specific warnings not covered by previous pragma
#pragma warning(                                                          \
    disable : 4244 4056 4305 4800 4267 4996 4756 4661 4385 4101)

#define __func__ __FUNCTION__
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif /* _CRT_SECURE_NO_WARNINGS */
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#include <direct.h>
#include <io.h>
#include <windows.h>
#define WB_USE_WINDOWS
#else /* _MSC_VER */
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define WB_USE_UNIX
#ifdef __APPLE__
#include <mach/mach_time.h>
#define WB_USE_DARWIN
#else /* __APPLE__ */
#define WB_USE_LINUX
#endif /* __APPLE__ */
#endif /* _MSC_VER */

#define wbStmt(stmt) stmt

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#define wbLine __LINE__
#define wbFile __FILE__
#define wbFunction __func__

#define wbExit()                                                          \
  wbAssert(0);                                                            \
  exit(1)

#ifdef WB_USE_COURSERA
#define wbLogger_printOnExit 1
#else /* WB_USE_COURSERA */
#define wbLogger_printOnLog 1
#endif /* WB_USE_COURSERA */

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#ifdef __cplusplus
#define EXTERN_C extern "C"
#define START_EXTERN_C EXTERN_C {
#define END_EXTERN_C }
#else
#define EXTERN_C
#define START_EXTERN_C
#define END_EXTERN_C
#endif /* __cplusplus */

/***********************************************************/
/***********************************************************/
/***********************************************************/

#include "vendor/json11.hpp"
#define WB_USE_JSON11 1

/***********************************************************/
/***********************************************************/
/***********************************************************/

#define LAZY_FILE_LOAD
extern char *solutionJSON;

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#ifdef WB_USE_OPENCL
#ifdef WB_USE_DARWIN
#include <OpenCL/opencl.h>
#else /* WB_USE_DARWIN */
#include <CL/cl.h>
#endif /* WB_USE_DARWIN */
#endif /* WB_USE_OPENCL */

#include "wbTypes.h"

#include "wbAssert.h"
#include "wbMalloc.h"
#include "wbString.h"
#include "wbUtils.h"

#include "wbArg.h"
#include "wbCUDA.h"
#include "wbCast.h"
#include "wbComparator.h"
#include "wbDirectory.h"
#include "wbExit.h"
#include "wbExport.h"
#include "wbFile.h"
#include "wbImage.h"
#include "wbImport.h"
#include "wbInit.h"
#include "wbLogger.h"
#include "wbMD5.h"
#include "wbMPI.h"
#include "wbSolution.h"
#include "wbSparse.h"
#include "wbThrust.h"
#include "wbTimer.h"
#include "wbPath.h"

#include "wbDataset.h"

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#endif /* __WB_H__ */
