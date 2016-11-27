

#include "wb.h"

static inline void wbImportCSV_setFile(wbImportCSV_t csv,
                                       const char *path) {
  if (csv != nullptr) {
    if (wbImportCSV_getFile(csv) != nullptr) {
      wbFile_delete(wbImportCSV_getFile(csv));
    }
    if (path != nullptr) {
      wbImportCSV_getFile(csv) = wbFile_open(path, "r");
    } else {
      wbImportCSV_getFile(csv) = nullptr;
    }
  }

  return;
}

static inline wbImportCSV_t wbImportCSV_new(void) {
  wbImportCSV_t csv;

  csv = wbNew(struct st_wbImportCSV_t);

  wbImportCSV_setRowCount(csv, -1);
  wbImportCSV_setColumnCount(csv, -1);
  wbImportCSV_setData(csv, NULL);
  wbImportCSV_getFile(csv) = nullptr;
  wbImportCSV_setSeperator(csv, '\0');

  return csv;
}

static inline void wbImportCSV_delete(wbImportCSV_t csv) {
  if (csv != nullptr) {
    wbImportCSV_setFile(csv, NULL);
    if (wbImportCSV_getData(csv)) {
      wbDelete(wbImportCSV_getData(csv));
    }
    wbDelete(csv);
  }
}

static inline wbImportCSV_t wbImportCSV_findDimensions(wbImportCSV_t csv,
                                                       int *resRows,
                                                       int *resColumns) {
  int rows = 0, columns = -1;
  char *line;
  wbFile_t file;
  char seperator[2];

  if (csv == nullptr) {
    return NULL;
  }

  if (wbImportCSV_getSeperator(csv) == '\0') {
    seperator[0] = ',';
  } else {
    seperator[0] = wbImportCSV_getSeperator(csv);
  }
  seperator[1] = '\0';

  file = wbImportCSV_getFile(csv);

  while ((line = wbFile_readLine(file)) != nullptr) {
    int currColumn = 0;
    char *token    = strtok(line, seperator);
    while (token != nullptr) {
      token = strtok(NULL, seperator);
      currColumn++;
    }
    rows++;
    if (columns == -1) {
      columns = currColumn;
    }
    if (columns != currColumn) {
      wbLog(ERROR, "The csv file is not rectangular.");
    }
    wbAssert(columns == currColumn);
  }

  wbFile_rewind(file);

  *resRows    = rows;
  *resColumns = columns;

  return csv;
}

static inline int *csv_readAsInteger(wbFile_t file, char sep, int rows,
                                     int columns) {
  int ii = 0;
  int *data;
  char *line;
  int var;
  char seperator[2];

  if (file == nullptr) {
    return NULL;
  }

  data = wbNewArray(int, rows *columns);

  if (sep == '\0') {
    seperator[0] = ',';
  } else {
    seperator[0] = sep;
  }
  seperator[1] = '\0';

  // printf("cols = %d rows = %d\n", columns, rows);
  if (columns == 1) {
    while ((line = wbFile_readLine(file)) != nullptr) {
      sscanf(line, "%d", &var);
      // printf("reading %d\n", var);
      data[ii++] = var;
    }
  } else {
    while ((line = wbFile_readLine(file)) != nullptr) {
      char *token = strtok(line, seperator);
      while (token != nullptr) {
        sscanf(token, "%d", &var);
        token      = strtok(NULL, seperator);
        data[ii++] = var;
      }
    }
  }

  return data;
}

static inline wbReal_t *csv_readAsReal(wbFile_t file, char sep, int rows,
                                       int columns) {
  int ii = 0;
  wbReal_t *data;
  char *line;
  wbReal_t var;
  char seperator[2];

  if (file == nullptr) {
    return NULL;
  }

  data = wbNewArray(wbReal_t, rows * columns);

  if (sep == '\0') {
    seperator[0] = ',';
  } else {
    seperator[0] = sep;
  }
  seperator[1] = '\0';

  if (columns == 1) {
    while ((line = wbFile_readLine(file)) != nullptr) {
      sscanf(line, "%f", &var);
      data[ii++] = var;
    }
  } else {
    while ((line = wbFile_readLine(file)) != nullptr) {
      char *token = strtok(line, seperator);
      while (token != nullptr) {
        sscanf(token, "%f", &var);
        token      = strtok(NULL, seperator);
        data[ii++] = var;
      }
    }
  }

  return data;
}

static inline wbImportCSV_t wbImportCSV_read(wbImportCSV_t csv,
                                             wbType_t type) {
  void *data;
  wbFile_t file;
  char seperator;
  int rows, columns;

  if (csv == nullptr) {
    return NULL;
  }

  if (wbImportCSV_getRowCount(csv) == -1 ||
      wbImportCSV_getColumnCount(csv) == -1) {
    if (wbImportCSV_findDimensions(csv, &rows, &columns) == nullptr) {
      wbLog(ERROR, "Failed to figure out csv dimensions.");
      return NULL;
    }
    wbImportCSV_setRowCount(csv, rows);
    wbImportCSV_setColumnCount(csv, columns);
  }

  file      = wbImportCSV_getFile(csv);
  seperator = wbImportCSV_getSeperator(csv);
  rows      = wbImportCSV_getRowCount(csv);
  columns   = wbImportCSV_getColumnCount(csv);

  if (wbImportCSV_getData(csv) != nullptr) {
    wbDelete(wbImportCSV_getData(csv));
    wbImportCSV_setData(csv, NULL);
  }

  if (type == wbType_integer) {
    // printf("ReadXXXing as integer...\n");
    data = csv_readAsInteger(file, seperator, rows, columns);
  } else {
    data = csv_readAsReal(file, seperator, rows, columns);
  }

  wbImportCSV_setData(csv, data);

  return csv;
}

static inline wbImportCSV_t wbImportCSV_readAsInteger(wbImportCSV_t csv) {
  return wbImportCSV_read(csv, wbType_integer);
}

static inline wbImportCSV_t wbImportCSV_readAsReal(wbImportCSV_t csv) {
  return wbImportCSV_read(csv, wbType_real);
}

static inline void wbImportRaw_setFile(wbImportRaw_t raw,
                                       const char *path) {
  if (raw != nullptr) {
    if (wbImportRaw_getFile(raw) != nullptr) {
      wbFile_delete(wbImportRaw_getFile(raw));
    }
    if (path != nullptr) {
      wbImportRaw_getFile(raw) = wbFile_open(path, "r");
    } else {
      wbImportRaw_getFile(raw) = nullptr;
    }
  }

  return;
}

static inline wbImportRaw_t wbImportRaw_new(void) {
  wbImportRaw_t raw;

  raw = wbNew(struct st_wbImportRaw_t);

  wbImportRaw_setRowCount(raw, -1);
  wbImportRaw_setColumnCount(raw, -1);
  wbImportRaw_setData(raw, NULL);
  wbImportRaw_getFile(raw) = nullptr;

  return raw;
}

static inline void wbImportRaw_delete(wbImportRaw_t raw) {
  if (raw != nullptr) {
    wbImportRaw_setFile(raw, NULL);
    if (wbImportRaw_getData(raw)) {
      wbDelete(wbImportRaw_getData(raw));
    }
    wbDelete(raw);
  }
}

static inline wbBool lineHasSpace(const char *line) {
  while (*line != '\0') {
    if (*line == ' ') {
      return wbTrue;
    }
    line++;
  }
  return wbFalse;
}

static inline char *lineStrip(const char *line) {
  char *sl    = wbString_duplicate(line);
  char *iter  = sl;
  size_t slen = strlen(line);

  iter += slen - 1;
  while (*iter == '\0' || *iter == '\r' || *iter == '\t' ||
         *iter == '\n' || *iter == ' ') {
    *iter-- = '\0';
  }
  return sl;
}

static inline wbBool wbImportRaw_findDimensions(wbImportRaw_t raw) {
  if (raw != nullptr) {
    int rows;
    int columns;
    char *line;
    wbFile_t file;
    char *strippedLine;

    file = wbImportRaw_getFile(raw);

    wbFile_rewind(file);

    line = wbFile_readLine(file);

    if (line == nullptr) {
      return wbTrue;
    }

    strippedLine = lineStrip(line);

    if (lineHasSpace(strippedLine)) {
      sscanf(strippedLine, "%d %d", &rows, &columns);
    } else {
      columns = 1;
      sscanf(strippedLine, "%d", &rows);
    }

    wbImportRaw_setRowCount(raw, rows);
    wbImportRaw_setColumnCount(raw, columns);

    wbDelete(strippedLine);

    return wbFalse;
  }

  return wbTrue;
}

static inline wbImportRaw_t wbImportRaw_read(wbImportRaw_t raw,
                                             wbType_t type) {
  void *data;
  wbFile_t file;
  char seperator;
  int rows, columns;

  if (raw == nullptr) {
    return NULL;
  }

  if (wbImportRaw_getRowCount(raw) == -1 ||
      wbImportRaw_getColumnCount(raw) == -1) {
    if (wbImportRaw_findDimensions(raw)) {
      wbLog(ERROR, "Failed to figure out raw dimensions.");
      return NULL;
    }
  }

  file      = wbImportRaw_getFile(raw);
  seperator = ' ';
  rows      = wbImportRaw_getRowCount(raw);
  columns   = wbImportRaw_getColumnCount(raw);

  if (wbImportRaw_getData(raw) != nullptr) {
    wbDelete(wbImportRaw_getData(raw));
    wbImportRaw_setData(raw, NULL);
  }

  if (type == wbType_integer) {
    // printf("Rdin gas integer...\n");
    data = csv_readAsInteger(file, seperator, rows, columns);
  } else {
    data = csv_readAsReal(file, seperator, rows, columns);
  }

  wbImportRaw_setData(raw, data);

  return raw;
}

static inline wbImportRaw_t wbImportRaw_readAsInteger(wbImportRaw_t raw) {
  return wbImportRaw_read(raw, wbType_integer);
}

static inline wbImportRaw_t wbImportRaw_readAsReal(wbImportRaw_t raw) {
  return wbImportRaw_read(raw, wbType_real);
}

static inline wbImportText_t wbImportText_new(void) {
  wbImportText_t text;

  text = wbNew(struct st_wbImportText_t);

  wbImportText_setLength(text, 0);
  wbImportText_setData(text, NULL);
  wbImportText_getFile(text) = nullptr;

  return text;
}

static inline void wbImportText_setFile(wbImportText_t text,
                                        const char *path) {
  if (text != nullptr) {
    if (wbImportText_getFile(text) != nullptr) {
      wbFile_delete(wbImportText_getFile(text));
    }
    if (path != nullptr) {
      wbImportText_getFile(text) = wbFile_open(path, "r");
    } else {
      wbImportText_getFile(text) = nullptr;
    }
  }

  return;
}

static inline void wbImportText_delete(wbImportText_t text) {
  if (text != nullptr) {
    wbImportText_setFile(text, NULL);
    if (wbImportText_getData(text)) {
      wbDelete(wbImportText_getData(text));
    }
    wbDelete(text);
  }
}

static inline wbImportText_t wbImportText_read(wbImportText_t text) {
  char *data;
  wbFile_t file;
  int length;

  if (text == nullptr) {
    return NULL;
  }

  file = wbImportText_getFile(text);

  if (wbImportText_getData(text) != nullptr) {
    wbDelete(wbImportText_getData(text));
    wbImportText_setData(text, NULL);
  }

  length = wbFile_size(file);
  data   = wbFile_read(file, length);

  wbImportText_setData(text, data);
  wbImportText_setLength(text, length);

  return text;
}

static inline wbImport_t wbImport_open(const char *file,
                                       wbImportKind_t kind) {
  wbImport_t imp;

  if (file == nullptr) {
    wbLog(ERROR, "Go NULL for file value.");
    wbExit();
  }

  if (!wbFile_existsQ(file)) {
    wbLog(ERROR, "File ", file, " does not exist.");
    wbExit();
  }

  wbImport_setKind(imp, kind);

  if (kind == wbImportKind_raw) {
    wbImportRaw_t raw = wbImportRaw_new();
    wbImportRaw_setFile(raw, file);
    wbImport_setRaw(imp, raw);
  } else if (kind == wbImportKind_tsv || kind == wbImportKind_csv) {
    wbImportCSV_t csv = wbImportCSV_new();
    if (kind == wbImportKind_csv) {
      wbImportCSV_setSeperator(csv, ',');
    } else {
      wbImportCSV_setSeperator(csv, '\t');
    }
    wbImportCSV_setFile(csv, file);
    wbImport_setCSV(imp, csv);
  } else if (kind == wbImportKind_text) {
    wbImportText_t text = wbImportText_new();
    wbImportText_setFile(text, file);
    wbImport_setText(imp, text);
  } else if (kind == wbImportKind_ppm) {
    wbImage_t img = wbPPM_import(file);
    wbImport_setImage(imp, img);
  } else {
    wbLog(ERROR, "Invalid import type.");
    wbExit();
  }

  return imp;
}

static inline wbImport_t wbImport_open(const char *file,
                                       const char *type0) {
  wbImport_t imp;
  wbImportKind_t kind;
  char *type;

  type = wbString_toLower(type0);

  if (wbString_sameQ(type, "csv")) {
    kind = wbImportKind_csv;
  } else if (wbString_sameQ(type, "tsv")) {
    kind = wbImportKind_tsv;
  } else if (wbString_sameQ(type, "raw") || wbString_sameQ(type, "dat")) {
    kind = wbImportKind_raw;
  } else if (wbString_sameQ(type, "ppm")) {
    kind = wbImportKind_ppm;
  } else if (wbString_sameQ(type, "text") || wbString_sameQ(type, "txt")) {
    kind = wbImportKind_text;
  } else {
    wbLog(ERROR, "Invalid import type ", type0);
    wbExit();
  }

  imp = wbImport_open(file, kind);

  wbDelete(type);

  return imp;
}

static inline void wbImport_close(wbImport_t imp) {
  wbImportKind_t kind;

  kind = wbImport_getKind(imp);
  if (kind == wbImportKind_tsv || kind == wbImportKind_csv) {
    wbImportCSV_t csv = wbImport_getCSV(imp);
    wbImportCSV_delete(csv);
    wbImport_setCSV(imp, NULL);
  } else if (kind == wbImportKind_raw) {
    wbImportRaw_t raw = wbImport_getRaw(imp);
    wbImportRaw_delete(raw);
    wbImport_setRaw(imp, NULL);
  } else if (kind == wbImportKind_text) {
    wbImportText_t text = wbImport_getText(imp);
    wbImportText_delete(text);
    wbImport_setText(imp, NULL);
  } else if (kind == wbImportKind_ppm) {
  } else {
    wbLog(ERROR, "Invalid import type.");
    wbExit();
  }
  return;
}

static inline void *wbImport_read(wbImport_t imp, wbType_t type) {
  void *data = nullptr;
  wbImportKind_t kind;

  kind = wbImport_getKind(imp);
  if (kind == wbImportKind_tsv || kind == wbImportKind_csv) {
    wbImportCSV_t csv = wbImport_getCSV(imp);
    wbImportCSV_read(csv, type);
    data = wbImportCSV_getData(csv);
  } else if (kind == wbImportKind_raw) {
    wbImportRaw_t raw = wbImport_getRaw(imp);
    wbImportRaw_read(raw, type);
    data = wbImportRaw_getData(raw);
  } else if (wbImportKind_text == kind) {
    wbImportText_t text = wbImport_getText(imp);
    text                = wbImportText_read(text);
    data                = wbImportText_getData(text);

  } else {
    wbLog(ERROR, "Invalid import type.");
    wbExit();
  }
  return data;
}

static inline int *wbImport_readAsInteger(wbImport_t imp) {
  void *data = wbImport_read(imp, wbType_integer);
  return (int *)data;
}

static inline wbReal_t *wbImport_readAsReal(wbImport_t imp) {
  void *data = wbImport_read(imp, wbType_real);
  return (wbReal_t *)data;
}

static inline wbChar_t *wbImport_readAsText(wbImport_t imp) {
  void *data = wbImport_read(imp, wbType_ubit8);
  return (wbChar_t *)data;
}

static wbImportKind_t _parseImportExtension(const char *file) {
  char *extension;
  wbImportKind_t kind;

  extension = wbFile_extension(file);

  if (wbString_sameQ(extension, "csv")) {
    kind = wbImportKind_csv;
  } else if (wbString_sameQ(extension, "tsv")) {
    kind = wbImportKind_tsv;
  } else if (wbString_sameQ(extension, "raw") ||
             wbString_sameQ(extension, "dat")) {
    kind = wbImportKind_raw;
  } else if (wbString_sameQ(extension, "ppm") ||
             wbString_sameQ(extension, "pbm")) {
    kind = wbImportKind_ppm;
  } else if (wbString_sameQ(extension, "text") ||
             wbString_sameQ(extension, "txt")) {
    kind = wbImportKind_text;
  } else {
    kind = wbImportKind_unknown;
    wbLog(ERROR, "File ", file, " does not have a compatible extension.");
  }

  wbDelete(extension);

  return kind;
}

void *wbImport(const char *file, int *resRows, int *resColumns,
               const char *type) {
  void *data, *res;
  wbImport_t imp;
  size_t sz;
  int columns = 0, rows = 0;
  wbImportKind_t kind;

  if (file == nullptr) {
    fprintf(stderr, "Failed to import file.\n");
    wbExit();
  }

  kind = _parseImportExtension(file);

  wbAssert(kind != wbImportKind_unknown);

  imp = wbImport_open(file, kind);
  if (wbString_sameQ(type, "Real")) {
    data = wbImport_readAsReal(imp);
    sz   = sizeof(wbReal_t);
  } else if (wbString_sameQ(type, "Text")) {
    data = wbImport_readAsText(imp);
    sz   = sizeof(char);
  } else {
    // printf("Reading as integer..d\n");
    data = wbImport_readAsInteger(imp);
    sz   = sizeof(int);
  }

  if (kind == wbImportKind_csv || kind == wbImportKind_tsv) {
    rows    = wbImportCSV_getRowCount(wbImport_getCSV(imp));
    columns = wbImportCSV_getColumnCount(wbImport_getCSV(imp));
  } else if (kind == wbImportKind_raw) {
    rows    = wbImportRaw_getRowCount(wbImport_getRaw(imp));
    columns = wbImportRaw_getColumnCount(wbImport_getRaw(imp));
  } else if (kind == wbImportKind_text) {
    rows    = 1;
    columns = wbImportText_getLength(wbImport_getText(imp));
  }

  if (rows == 1 && columns > 0) {
    rows    = columns;
    columns = 1;
  }

  if (resRows != nullptr) {
    *resRows = rows;
  }

  if (resColumns != nullptr) {
    *resColumns = columns;
  }

  res = wbMalloc(sz * rows * columns);
  memcpy(res, data, sz * rows * columns);

  wbImport_close(imp);

  return res;
}

void *wbImport(const char *file, int *rows, int *columns) {
  return wbImport(file, rows, columns, "Real");
}

EXTERN_C void *wbImport(const char *file, int *rows) {
  return wbImport(file, rows, NULL, "Real");
}

void *wbImport(const char *file, int *res_rows, const char *type) {
  int cols, rows;
  void *res = wbImport(file, &rows, &cols, type);
  if (rows == 1 && cols > 1) {
    rows = cols;
  }
  *res_rows = rows;
  return res;
}

wbImage_t wbImport(const char *file) {
  wbImage_t img;
  wbImport_t imp;
  wbImportKind_t kind;

  if (file == nullptr) {
    fprintf(stderr, "Failed to import file.\n");
    wbExit();
  }

  kind = _parseImportExtension(file);

  wbAssert(kind == wbImportKind_ppm);

  imp = wbImport_open(file, kind);
  img = wbImport_getImage(imp);
  wbImport_close(imp);

  return img;
}

int wbImport_flag(const char *file) {
  int res;
  wbFile_t fh      = wbFile_open(file, "r");
  const char *line = wbFile_readLine(fh);
  sscanf(line, "%d", &res);
  wbFile_close(fh);
  return res;
}
