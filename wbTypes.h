

#ifndef __WB_TYPES_H__
#define __WB_TYPES_H__

#include "wbAssert.h"

typedef bool wbBool;
typedef float wbReal_t;
typedef char wbChar_t;

typedef struct st_wbTimerNode_t *wbTimerNode_t;
typedef struct st_wbTimer_t *wbTimer_t;
typedef struct st_wbLogEntry_t *wbLogEntry_t;
typedef struct st_wbLogger_t *wbLogger_t;
typedef struct st_wbArg_t wbArg_t;
typedef struct st_wbImage_t *wbImage_t;
typedef struct st_wbFile_t *wbFile_t;

#define wbTrue true
#define wbFalse false

typedef enum en_wbType_t {
  wbType_unknown = -1,
  wbType_ascii   = 1,
  wbType_bit8,
  wbType_ubit8,
  wbType_integer,
  wbType_float,
  wbType_double
} wbType_t;

static inline size_t wbType_size(wbType_t ty) {
  switch (ty) {
    case wbType_unknown:
      wbAssert(false && "Invalid wbType_unknown");
      return 0;
    case wbType_ascii:
      return sizeof(char);
    case wbType_bit8:
      return sizeof(char);
    case wbType_ubit8:
      return sizeof(unsigned char);
    case wbType_integer:
      return sizeof(int);
    case wbType_float:
      return sizeof(float);
    case wbType_double:
      return sizeof(double);
  }
  wbAssert(false && "Invalid type");
  return 0;
}

#endif /* __WB_TYPES_H__ */
