
#include "wb.h"

static inline void wbExportText_setFile(wbExportText_t text,
                                        const char *path) {
  if (text != nullptr) {
    if (wbExportText_getFile(text) != nullptr) {
      wbFile_delete(wbExportText_getFile(text));
    }
    if (path != nullptr) {
      wbExportText_getFile(text) = wbFile_open(path, "w+");
    } else {
      wbExportText_getFile(text) = nullptr;
    }
  }

  return;
}

static inline wbExportText_t wbExportText_new(void) {
  wbExportText_t text;

  text = wbNew(struct st_wbExportText_t);

  wbExportText_getFile(text) = nullptr;
  wbExportText_setLength(text, -1);

  return text;
}

static inline void wbExportText_delete(wbExportText_t text) {
  if (text != nullptr) {
    wbExportText_setFile(text, NULL);
    wbDelete(text);
  }
  return;
}

static inline void wbExportText_write(wbExportText_t text,
                                      const char *data, int length) {
  int ii;
  FILE *handle;
  wbFile_t file;

  if (text == nullptr || wbExportText_getFile(text) == nullptr) {
    return;
  }

  file = wbExportText_getFile(text);

  handle = wbFile_getFileHandle(file);

  if (handle == nullptr) {
    return;
  }

  for (ii = 0; ii < length; ii++) {
    fprintf(handle, "%c", data[ii]);
  }

  return;
}

static inline void wbExportRaw_setFile(wbExportRaw_t raw,
                                       const char *path) {
  if (raw != nullptr) {
    if (wbExportRaw_getFile(raw) != nullptr) {
      wbFile_delete(wbExportRaw_getFile(raw));
    }
    if (path != nullptr) {
      wbExportRaw_getFile(raw) = wbFile_open(path, "w+");
    } else {
      wbExportRaw_getFile(raw) = nullptr;
    }
  }

  return;
}

static inline wbExportRaw_t wbExportRaw_new(void) {
  wbExportRaw_t raw;

  raw = wbNew(struct st_wbExportRaw_t);

  wbExportRaw_getFile(raw) = nullptr;
  wbExportRaw_setRowCount(raw, -1);
  wbExportRaw_setColumnCount(raw, -1);

  return raw;
}

static inline void wbExportRaw_delete(wbExportRaw_t raw) {
  if (raw != nullptr) {
    wbExportRaw_setFile(raw, NULL);
    wbDelete(raw);
  }
  return;
}

static inline void wbExportRaw_write(wbExportRaw_t raw, void *data,
                                     int rows, int columns,
                                     wbType_t type) {
  int ii, jj;
  FILE *handle;
  wbFile_t file;

  if (raw == nullptr || wbExportRaw_getFile(raw) == nullptr) {
    return;
  }

  file = wbExportRaw_getFile(raw);

  handle = wbFile_getFileHandle(file);

  if (handle == nullptr) {
    return;
  }

  if (columns == 1) {
    fprintf(handle, "%d\n", rows);
  } else {
    fprintf(handle, "%d %d\n", rows, columns);
  }

  for (ii = 0; ii < rows; ii++) {
    for (jj = 0; jj < columns; jj++) {
      if (type == wbType_integer) {
        int elem = ((int *)data)[ii * columns + jj];
        fprintf(handle, "%d", elem);
      } else if (type == wbType_ubit8) {
        int elem = ((unsigned char *)data)[ii * columns + jj];
        fprintf(handle, "%d", elem);
      } else {
        wbReal_t elem = ((wbReal_t *)data)[ii * columns + jj];
        fprintf(handle, "%f", elem);
      }
      if (jj == columns - 1) {
        fprintf(handle, "\n");
      } else {
        fprintf(handle, " ");
      }
    }
  }

  return;
}

static inline void wbExportCSV_setFile(wbExportCSV_t csv,
                                       const char *path) {
  if (csv != nullptr) {
    if (wbExportCSV_getFile(csv) != nullptr) {
      wbFile_delete(wbExportCSV_getFile(csv));
    }
    if (path != nullptr) {
      wbExportCSV_getFile(csv) = wbFile_open(path, "w+");
    } else {
      wbExportCSV_getFile(csv) = nullptr;
    }
  }

  return;
}

static inline wbExportCSV_t wbExportCSV_new(void) {
  wbExportCSV_t csv;

  csv = wbNew(struct st_wbExportCSV_t);

  wbExportCSV_getFile(csv) = nullptr;
  wbExportCSV_setColumnCount(csv, -1);
  wbExportCSV_setRowCount(csv, -1);
  wbExportCSV_setSeperator(csv, '\0');

  return csv;
}

static inline void wbExportCSV_delete(wbExportCSV_t csv) {
  if (csv != nullptr) {
    wbExportCSV_setFile(csv, NULL);
    wbDelete(csv);
  }
}

static inline void wbExportCSV_write(wbExportCSV_t csv, void *data,
                                     int rows, int columns, char sep,
                                     wbType_t type) {
  int ii, jj;
  wbFile_t file;
  FILE *handle;
  char seperator[2];

  if (csv == nullptr || wbExportCSV_getFile(csv) == nullptr) {
    return;
  }

  file = wbExportCSV_getFile(csv);

  handle = wbFile_getFileHandle(file);

  if (handle == nullptr) {
    return;
  }

  if (sep == '\0') {
    seperator[0] = ',';
  } else {
    seperator[0] = sep;
  }
  seperator[1] = '\0';

  for (ii = 0; ii < rows; ii++) {
    for (jj = 0; jj < columns; jj++) {
      if (type == wbType_integer) {
        int elem = ((int *)data)[ii * columns + jj];
        fprintf(handle, "%d", elem);
      } else if (type == wbType_ubit8) {
        int elem = ((unsigned char *)data)[ii * columns + jj];
        fprintf(handle, "%d", elem);
      } else {
        wbReal_t elem = ((wbReal_t *)data)[ii * columns + jj];
        fprintf(handle, "%f", elem);
      }
      if (jj == columns - 1) {
        fprintf(handle, "\n");
      } else {
        fprintf(handle, "%s", seperator);
      }
    }
  }

  return;
}

static inline wbExport_t wbExport_open(const char *file,
                                       wbExportKind_t kind) {
  wbExport_t exprt;

  if (file == nullptr) {
    wbLog(ERROR, "Go NULL for file value.");
    wbExit();
  }

  wbExport_setFile(exprt, NULL);
  wbExport_setKind(exprt, kind);

  if (kind == wbExportKind_raw) {
    wbExportRaw_t raw = wbExportRaw_new();
    wbExportRaw_setFile(raw, file);
    wbExport_setRaw(exprt, raw);
  } else if (kind == wbExportKind_text) {
    wbExportText_t txt = wbExportText_new();
    wbExportText_setFile(txt, file);
    wbExport_setText(exprt, txt);
  } else if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
    wbExportCSV_t csv = wbExportCSV_new();
    if (kind == wbExportKind_csv) {
      wbExportCSV_setSeperator(csv, ',');
    } else {
      wbExportCSV_setSeperator(csv, '\t');
    }
    wbExportCSV_setFile(csv, file);
    wbExport_setCSV(exprt, csv);
  } else if (kind == wbExportKind_ppm) {
    wbExport_setFile(exprt, wbString_duplicate(file));
  } else {
    wbLog(ERROR, "Invalid export type.");
    wbExit();
  }

  return exprt;
}

static inline wbExport_t wbExport_open(const char *file,
                                       const char *type0) {
  wbExport_t exprt;
  wbExportKind_t kind;
  char *type;

  type = wbString_toLower(type0);

  if (wbString_sameQ(type, "csv")) {
    kind = wbExportKind_csv;
  } else if (wbString_sameQ(type, "tsv")) {
    kind = wbExportKind_tsv;
  } else if (wbString_sameQ(type, "raw") || wbString_sameQ(type, "dat")) {
    kind = wbExportKind_raw;
  } else if (wbString_sameQ(type, "ppm") || wbString_sameQ(type, "pbm")) {
    kind = wbExportKind_ppm;
  } else if (wbString_sameQ(type, "txt") || wbString_sameQ(type, "text")) {
    kind = wbExportKind_text;
  } else {
    wbLog(ERROR, "Invalid export type ", type0);
    wbExit();
  }

  exprt = wbExport_open(file, kind);

  wbDelete(type);

  return exprt;
}

static inline void wbExport_close(wbExport_t exprt) {
  wbExportKind_t kind;

  kind = wbExport_getKind(exprt);

  if (wbExport_getFile(exprt)) {
    wbDelete(wbExport_getFile(exprt));
  }

  if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
    wbExportCSV_t csv = wbExport_getCSV(exprt);
    wbExportCSV_delete(csv);
    wbExport_setCSV(exprt, NULL);
  } else if (kind == wbExportKind_raw) {
    wbExportRaw_t raw = wbExport_getRaw(exprt);
    wbExportRaw_delete(raw);
    wbExport_setRaw(exprt, NULL);
  } else if (kind == wbExportKind_text) {
    wbExportText_t text = wbExport_getText(exprt);
    wbExportText_delete(text);
    wbExport_setText(exprt, NULL);
  } else if (kind == wbExportKind_ppm) {
  } else {
    wbLog(ERROR, "Invalid export type.");
    wbExit();
  }
  return;
}

static inline void wbExport_writeAsImage(wbExport_t exprt, wbImage_t img) {
  wbAssert(wbExport_getKind(exprt) == wbExportKind_ppm);

  wbPPM_export(wbExport_getFile(exprt), img);

  return;
}

static inline void wbExport_write(wbExport_t exprt, void *data, int rows,
                                  int columns, char sep, wbType_t type) {
  wbExportKind_t kind;

  kind = wbExport_getKind(exprt);
  if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
    wbExportCSV_t csv = wbExport_getCSV(exprt);
    wbExportCSV_write(csv, data, rows, columns, sep, type);
  } else if (kind == wbExportKind_raw) {
    wbExportRaw_t raw = wbExport_getRaw(exprt);
    wbExportRaw_write(raw, data, rows, columns, type);
  } else if (kind == wbExportKind_text) {
    wbExportText_t text = wbExport_getText(exprt);
    if (columns == 0) {
      columns = 1;
    }
    if (rows == 0) {
      rows = 1;
    }
    wbExportText_write(text, (const char *)data, rows * columns);
  } else {
    wbLog(ERROR, "Invalid export type.");
    wbExit();
  }
  return;
}

static inline void wbExport_write(wbExport_t exprt, void *data, int rows,
                                  int columns, wbType_t type) {
  wbExport_write(exprt, data, rows, columns, ',', type);
}

static wbExportKind_t _parseExportExtension(const char *file) {
  char *extension;
  wbExportKind_t kind;

  extension = wbFile_extension(file);

  if (wbString_sameQ(extension, "csv")) {
    kind = wbExportKind_csv;
  } else if (wbString_sameQ(extension, "tsv")) {
    kind = wbExportKind_tsv;
  } else if (wbString_sameQ(extension, "raw") ||
             wbString_sameQ(extension, "dat")) {
    kind = wbExportKind_raw;
  } else if (wbString_sameQ(extension, "text") ||
             wbString_sameQ(extension, "txt")) {
    kind = wbExportKind_text;
  } else if (wbString_sameQ(extension, "ppm") ||
             wbString_sameQ(extension, "pbm")) {
    kind = wbExportKind_ppm;
  } else {
    kind = wbExportKind_unknown;
    wbLog(ERROR, "File ", file, " does not have a compatible extension.");
  }

  wbDelete(extension);

  return kind;
}

static void wbExport(const char *file, void *data, int rows, int columns,
                     wbType_t type) {
  wbExportKind_t kind;
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  kind  = _parseExportExtension(file);
  exprt = wbExport_open(file, kind);

  wbExport_write(exprt, data, rows, columns, type);
  wbExport_close(exprt);
}

void wbExport(const char *file, unsigned char *data, int rows) {
  wbExport(file, data, rows, 1);
  return;
}

void wbExport(const char *file, int *data, int rows) {
  wbExport(file, data, rows, 1);
  return;
}

void wbExport(const char *file, wbReal_t *data, int rows) {
  wbExport(file, data, rows, 1);
  return;
}

void wbExport(const char *file, unsigned char *data, int rows,
              int columns) {
  wbExportKind_t kind;
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  kind  = _parseExportExtension(file);
  exprt = wbExport_open(file, kind);

  wbExport_write(exprt, data, rows, columns, wbType_ubit8);
  wbExport_close(exprt);
}

void wbExport(const char *file, int *data, int rows, int columns) {
  wbExportKind_t kind;
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  kind  = _parseExportExtension(file);
  exprt = wbExport_open(file, kind);

  wbExport_write(exprt, data, rows, columns, wbType_integer);
  wbExport_close(exprt);
}

void wbExport(const char *file, wbReal_t *data, int rows, int columns) {
  wbExportKind_t kind;
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  kind  = _parseExportExtension(file);
  exprt = wbExport_open(file, kind);

  wbExport_write(exprt, data, rows, columns, wbType_real);
  wbExport_close(exprt);
}

void wbExport(const char *file, wbExportKind_t kind, void *data, int rows,
              int columns, wbType_t type) {
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  exprt = wbExport_open(file, kind);

  wbExport_write(exprt, data, rows, columns, type);
  wbExport_close(exprt);
}

void wbExport(const char *file, wbImage_t img) {
  wbExportKind_t kind;
  wbExport_t exprt;

  if (file == nullptr) {
    return;
  }

  kind  = _parseExportExtension(file);
  exprt = wbExport_open(file, kind);

  wbAssert(kind == wbExportKind_ppm);

  wbExport_writeAsImage(exprt, img);
  wbExport_close(exprt);
}

void wbExport_text(const char *file, void *data, int length) {
  wbExport(file, wbExportKind_text, data, 1, length, wbType_ascii);
}
