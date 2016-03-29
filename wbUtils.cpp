
#include "wb.h"
#include "vendor/sole.hpp"

std::string uuid() {
  const auto u4 = sole::uuid4();
  return u4.str();
}