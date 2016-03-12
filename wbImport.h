
#ifndef __WB_IMPORT_H__
#define __WB_IMPORT_H__

#include "wbImage.h"

typedef enum en_wbImportKind_t {
  wbImportKind_unknown = -1,
  wbImportKind_raw     = 0x1000,
  wbImportKind_csv,
  wbImportKind_tsv,
  wbImportKind_ppm,
  wbImportKind_text
} wbImportKind_t;

#define wbType_real wbType_float

typedef struct st_wbImportCSV_t {
  int rows;
  int columns;
  void *data;
  wbFile_t file;
  char seperator;
} * wbImportCSV_t;

#define wbImportCSV_getRowCount(csv) ((csv)->rows)
#define wbImportCSV_getColumnCount(csv) ((csv)->columns)
#define wbImportCSV_getData(csv) ((csv)->data)
#define wbImportCSV_getFile(csv) ((csv)->file)
#define wbImportCSV_getSeperator(csv) ((csv)->seperator)

#define wbImportCSV_setRowCount(csv, val)                                 \
  (wbImportCSV_getRowCount(csv) = val)
#define wbImportCSV_setColumnCount(csv, val)                              \
  (wbImportCSV_getColumnCount(csv) = val)
#define wbImportCSV_setData(csv, val) (wbImportCSV_getData(csv) = val)
#define wbImportCSV_setSeperator(csv, val)                                \
  (wbImportCSV_getSeperator(csv) = val)

typedef struct st_wbImportRaw_t {
  int rows;
  int columns;
  void *data;
  wbFile_t file;
} * wbImportRaw_t;

#define wbImportRaw_getRowCount(raw) ((raw)->rows)
#define wbImportRaw_getColumnCount(raw) ((raw)->columns)
#define wbImportRaw_getData(raw) ((raw)->data)
#define wbImportRaw_getFile(raw) ((raw)->file)

#define wbImportRaw_setRowCount(raw, val)                                 \
  (wbImportRaw_getRowCount(raw) = val)
#define wbImportRaw_setColumnCount(raw, val)                              \
  (wbImportRaw_getColumnCount(raw) = val)
#define wbImportRaw_setData(raw, val) (wbImportRaw_getData(raw) = val)

typedef struct st_wbImportText_t {
  int length;
  char *data;
  wbFile_t file;
} * wbImportText_t;

#define wbImportText_getLength(txt) ((txt)->length)
#define wbImportText_getData(txt) ((txt)->data)
#define wbImportText_getFile(txt) ((txt)->file)

#define wbImportText_setLength(txt, val)                                  \
  (wbImportText_getLength(txt) = val)
#define wbImportText_setData(txt, val) (wbImportText_getData(txt) = val)

typedef struct st_wbImport_t {
  wbImportKind_t kind;
  union {
    wbImportRaw_t raw;
    wbImportCSV_t csv;
    wbImportText_t text;
    wbImage_t img;
  } container;
} wbImport_t;

#define wbImport_getKind(imp) ((imp).kind)
#define wbImport_getContainer(imp) ((imp).container)
#define wbImport_getRaw(imp) (wbImport_getContainer(imp).raw)
#define wbImport_getCSV(imp) (wbImport_getContainer(imp).csv)
#define wbImport_getText(imp) (wbImport_getContainer(imp).text)
#define wbImport_getImage(imp) (wbImport_getContainer(imp).img)

#define wbImport_setKind(imp, val) (wbImport_getKind(imp) = val)
#define wbImport_setRaw(imp, val) (wbImport_getRaw(imp) = val)
#define wbImport_setCSV(imp, val) (wbImport_getCSV(imp) = val)
#define wbImport_setText(imp, val) (wbImport_getText(imp) = val)
#define wbImport_setImage(imp, val) (wbImport_getImage(imp) = val)

EXTERN_C void *wbImport(const char *file, int *rows);
void *wbImport(const char *file, int *rows, int *columns);
void *wbImport(const char *file, int *rows, const char *type);
void *wbImport(const char *file, int *resRows, int *resColumns,
               const char *type);
wbImage_t wbImport(const char *file);
int wbImport_flag(const char *file);

#endif /* __WB_IMPORT_H__ */
