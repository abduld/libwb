#ifndef __WB_UTILS_H__
#define __WB_UTILS_H__

#include "vendor/sole.hpp"

#ifdef WB_DEBUG
#define DEBUG(...) __VA_ARGS__
#else /* WB_DEBUG */
#define DEBUG(...)
#endif /* WB_DEBUG */

static std::string uuid() {
  auto u4 = sole::uuid4();
  return u4.str();
}

#endif /* __WB_UTILS_H__ */
