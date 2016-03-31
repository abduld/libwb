#define CATCH_CONFIG_MAIN

#include "wb.h"
#include "vendor/catch.hpp"



TEST_CASE("Can use basic functions", "[WB]") {
  wbTime_start(GPU, "timer."); //@@ start a timer
  wbTime_stop(GPU, "timer."); //@@ stop the timer

}
