

#ifndef __WB_H__
#define __WB_H__

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _MSC_VER
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
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
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
/***********************************************************/
/***********************************************************/
/***********************************************************/

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

#include <wbTypes.h>

#include <wbAssert.h>
#include <wbMalloc.h>
#include <wbString.h>
#include <wbUtils.h>

#include <wbArg.h>
#include <wbCUDA.h>
#include <wbCast.h>
#include <wbComparator.h>
#include <wbDirectory.h>
#include <wbExit.h>
#include <wbExport.h>
#include <wbFile.h>
#include <wbImage.h>
#include <wbImport.h>
#include <wbInit.h>
#include <wbLogger.h>
#include <wbMD5.h>
#include <wbMPI.h>
#include <wbSolution.h>
#include <wbSparse.h>
#include <wbThrust.h>
#include <wbTimer.h>

#include <wbDataGenerator.h>

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#endif /* __WB_H__ */
