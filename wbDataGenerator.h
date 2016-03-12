#ifndef __WB_DATASET_GENERATOR__
#define __WB_DATASET_GENERATOR__

#include "wbImport.h"
#include "wbTypes.h"

typedef struct {
  int rows;
  int cols;
  wbType_t type;
  int minVal;
  int maxVal;
} wbCSV_GenerateParams_t;

typedef struct {
  int rows;
  int cols;
  wbType_t type;
  int minVal;
  int maxVal;
} wbTSV_GenerateParams_t;

typedef struct {
  int rows;
  int cols;
  int minVal;
  int maxVal;
  wbType_t type;
} wbRaw_GenerateParams_t;

typedef struct {
  int width;
  int height;
  int channels;
  float minVal;
  float maxVal;
} wbPPM_GenerateParams_t;

typedef struct { int length; } wbText_GenerateParams_t;

typedef union {
  wbCSV_GenerateParams_t csv;
  wbRaw_GenerateParams_t raw;
  wbTSV_GenerateParams_t tsv;
  wbPPM_GenerateParams_t ppm;
  wbText_GenerateParams_t text;
} wbGenerateParams_t;

EXTERN_C void GenerateDataset(const char *path, wbExportKind_t kind,
                              wbGenerateParams_t params);

#endif /* __WB_DATASET_GENERATOR__ */
