#ifndef __WB_UTILS_H__
#define __WB_UTILS_H__

#include "wbString.h"
#include "vendor/sole.hpp"

#ifdef WB_DEBUG
#define DEBUG(...) __VA_ARGS__
#else /* WB_DEBUG */
#define DEBUG(...)
#endif /* WB_DEBUG */

static char* uuid() {
  auto u4 = sole::uuid4();
  return wbString_duplicate(u4.str());
}

#endif /* __WB_UTILS_H__ */
