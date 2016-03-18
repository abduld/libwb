
#include "wb.h"
#include "vendor/catch.hpp"

TEST_CASE("Can create Raw dataset", "[DataGenerator]") {
  wbGenerateParams_t params;
  params.raw.rows   = 2;
  params.raw.cols   = 300;
  params.raw.minVal = 0;
  params.raw.maxVal = 30;
  params.raw.type   = wbType_integer;
  wbDataset_generate(
      wbPath_join(wbDirectory_current(), "test-dataset", "test.raw"),
      wbExportKind_raw, params);
}

TEST_CASE("Can create Text dataset", "[DataGenerator]") {
  wbGenerateParams_t params;
  params.text.length = 2000;
  wbDataset_generate(
      wbPath_join(wbDirectory_current(), "test-dataset", "test.txt"),
      wbExportKind_text, params);
}
