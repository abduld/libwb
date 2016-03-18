#include "wb.h"

template <typename T>
static inline T _min(const T &x, const T &y) {
  return x < y ? x : y;
}

template <typename T>
static inline T _max(const T &x, const T &y) {
  return x > y ? x : y;
}

template <typename T>
inline T lerp(const double &x, const T &start, const T &end) {
  return (1 - x) * start + x * end;
}

static inline void genRandom(void *trgt, wbType_t type, double minVal,
                             double maxVal) {
  const int span  = maxVal - minVal;
  const int r     = rand();
  const double rf = ((double)r) / ((double)RAND_MAX);
  switch (type) {
    case wbType_ascii:
      *((char *)trgt) = (r % span) + minVal; // random printable character;
      break;
    case wbType_bit8:
      *((char *)trgt) = lerp<char>(rf, minVal, maxVal);
      break;
    case wbType_ubit8:
      *((unsigned char *)trgt) = lerp<unsigned char>(rf, minVal, maxVal);
      break;
    case wbType_integer:
      *((int *)trgt) = lerp<int>(rf, minVal, maxVal);
      break;
    case wbType_float: {
      *((float *)trgt) = lerp<float>(rf, minVal, maxVal);
      break;
    }
    case wbType_double: {
      *((double *)trgt) = lerp<double>(rf, minVal, maxVal);
      break;
    }
    case wbType_unknown:
      wbAssert(false && "Invalid wbType_unknown");
      break;
  }
  return;
}

static inline void *genRandomList(wbType_t type, size_t len, double minVal,
                                  double maxVal) {
  size_t ii;
  void *data = wbNewArray(char, wbType_size(type) * len);
  switch (type) {
    case wbType_ascii:
    case wbType_bit8: {
      char *iter = (char *)data;
      for (ii = 0; ii < len; ii++) {
        genRandom(iter++, type, minVal, maxVal);
      }
      break;
    }
    case wbType_ubit8: {
      unsigned char *iter = (unsigned char *)data;
      for (ii = 0; ii < len; ii++) {
        genRandom(iter++, type, minVal, maxVal);
      }
      break;
    }
    case wbType_integer: {
      int *iter = (int *)data;
      for (ii = 0; ii < len; ii++) {
        genRandom(iter++, type, minVal, maxVal);
      }
      break;
    }
    case wbType_float: {
      float *iter = (float *)data;
      for (ii = 0; ii < len; ii++) {
        genRandom(iter++, type, minVal, maxVal);
      }
      break;
    }
    case wbType_double: {
      double *iter = (double *)data;
      for (ii = 0; ii < len; ii++) {
        genRandom(iter++, type, minVal, maxVal);
      }
      break;
    }
    case wbType_unknown:
      wbAssert(false && "Invalid wbType_unknown");
      break;
  }
  return data;
}

static void genRaw(const char *path, wbRaw_GenerateParams_t params) {
  int rows      = _max(1, params.rows);
  int cols      = _max(1, params.cols);
  double minVal = params.minVal;
  double maxVal = params.maxVal;
  wbType_t type = params.type;
  void *data    = genRandomList(type, rows * cols, minVal, maxVal);
  wbExport(path, wbExportKind_raw, data, rows, cols, type);
  wbDelete(data);
}

static void genCSV(const char *path, wbCSV_GenerateParams_t params) {
  int rows      = _max(1, params.rows);
  int cols      = _max(1, params.cols);
  double minVal = params.minVal;
  double maxVal = params.maxVal;
  wbType_t type = params.type;
  void *data    = genRandomList(type, rows * cols, minVal, maxVal);
  wbExport(path, wbExportKind_csv, data, rows, cols, type);
  wbDelete(data);
}

static void genTSV(const char *path, wbTSV_GenerateParams_t params) {
  int rows      = _max(1, params.rows);
  int cols      = _max(1, params.cols);
  double minVal = params.minVal;
  double maxVal = params.maxVal;
  wbType_t type = params.type;
  void *data    = genRandomList(type, rows * cols, minVal, maxVal);
  wbExport(path, wbExportKind_tsv, data, rows, cols, type);
  wbDelete(data);
}

static void genText(const char *path, wbText_GenerateParams_t params) {
  int length    = _max(1, params.length);
  wbType_t type = wbType_ascii;
  void *data    = genRandomList(type, length, 32, 128);
  wbExport(path, wbExportKind_text, data, length, 1, type);
  wbDelete(data);
}

static void genPPM(const char *path, wbPPM_GenerateParams_t params) {
  int width     = _max(1, params.width);
  int height    = _max(1, params.height);
  int channels  = _max(1, params.channels);
  double minVal = params.minVal;
  double maxVal = params.maxVal;
  wbType_t type = wbType_float;
  float *data   = (float *)genRandomList(type, width * height * channels,
                                       minVal, maxVal);
  wbImage_t img = wbImage_new(width, height, channels, data);
  wbExport(path, img);
  wbImage_delete(img);
}

EXTERN_C void wbDataset_generate(const char *path, wbExportKind_t kind,
                                 wbGenerateParams_t params) {
  wbDirectory_create(wbDirectory_name(path));

  switch (kind) {
    case wbExportKind_raw:
      genRaw(path, params.raw);
      break;
    case wbExportKind_csv:
      genCSV(path, params.csv);
      break;
    case wbExportKind_tsv:
      genTSV(path, params.tsv);
      break;
    case wbExportKind_ppm:
      genPPM(path, params.ppm);
      break;
    case wbExportKind_text:
      genText(path, params.text);
      break;
    default:
      wbAssert(false && "Invalid Export kind");
  }
}
