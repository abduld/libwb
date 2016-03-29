#ifndef __WB_UTILS_H__
#define __WB_UTILS_H__

#include <iostream>
#include "vendor/sole.hpp"

#ifdef WB_DEBUG
#define DEBUG(...) __VA_ARGS__
#else /* WB_DEBUG */
#define DEBUG(...)
#endif /* WB_DEBUG */

static std::string uuid() {
  const auto u4 = sole::uuid4();
  return u4.str();
}

static std::string _sessionId{};
static std::string sessionId() {
#ifdef WB_USE_UNIX
  if (_sessionId != "") {
    char *env = std::getenv("SESSION_ID");
    if (env) {
      _sessionId = env;
    }
  }
#endif /* WB_USE_UNIX */
  return _sessionId;
}

#endif /* __WB_UTILS_H__ */
