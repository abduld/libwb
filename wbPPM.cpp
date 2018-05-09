
#include <math.h>
#include "wb.h"

static inline float _min(float x, float y) {
  return x < y ? x : y;
}

static inline float _max(float x, float y) {
  return x > y ? x : y;
}

static inline float _clamp(float x, float start, float end) {
  return _min(_max(x, start), end);
}

static const char *skipSpaces(const char *line) {
  while (*line == ' ' || *line == '\t') {
    line++;
    if (*line == '\0') {
      break;
    }
  }
  return line;
}

static char nextNonSpaceChar(const char *line0) {
  const char *line = skipSpaces(line0);
  return *line;
}

static wbBool isComment(const char *line) {
  char nextChar = nextNonSpaceChar(line);
  if (nextChar == '\0') {
    return wbTrue;
  } else {
    return nextChar == '#';
  }
}

static void parseDimensions(const char *line0, int *width, int *height) {
  const char *line = skipSpaces(line0);
  sscanf(line, "%d %d", width, height);
}

static void parseDimensions(const char *line0, int *width, int *height,
                            int *channels) {
  const char *line = skipSpaces(line0);
  sscanf(line, "%d %d %d", width, height, channels);
}

static void parseDepth(const char *line0, int *depth) {
  const char *line = skipSpaces(line0);
  sscanf(line, "%d", depth);
}

static char *nextLine(wbFile_t file) {
  char *line = nullptr;
  while ((line = wbFile_readLine(file)) != nullptr) {
    if (!isComment(line)) {
      break;
    }
  }
  return line;
}

wbImage_t wbPPM_import(const char *filename) {
  wbImage_t img;
  wbFile_t file;
  char *header;
  char *line;
  int ii, jj, kk, channels;
  int width, height, depth;
  unsigned char *charData, *charIter;
  float *imgData, *floatIter;
  float scale;

  img = nullptr;

  file = wbFile_open(filename, "rb");
  if (file == nullptr) {
    printf("Could not open %s\n", filename);
    goto cleanup;
  }

  header = wbFile_readLine(file);
  if (header == nullptr) {
    printf("Could not read from %s\n", filename);
    goto cleanup;
  } else if (strcmp(header, "P6") != 0 && strcmp(header, "P6\n") != 0 &&
             strcmp(header, "P5") != 0 && strcmp(header, "P5\n") != 0 &&
             strcmp(header, "S6") != 0 && strcmp(header, "S6\n") != 0) {
    printf("Could not find magic number for %s\n", filename);
    goto cleanup;
  }

  // P5 are monochrome while P6/S6 are rgb
  // S6 needs to parse number of channels out of file
  if (strcmp(header, "P5") == 0 || strcmp(header, "P5\n") == 0) {
    channels = 1;
    line     = nextLine(file);
    parseDimensions(line, &width, &height);
  } else if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
    channels = 3;
    line     = nextLine(file);
    parseDimensions(line, &width, &height);
  } else {
    line = nextLine(file);
    parseDimensions(line, &width, &height, &channels);
  }

  // the line now contains the depth information
  line = nextLine(file);
  parseDepth(line, &depth);

  // the rest of the lines contain the data in binary format
  charData = (unsigned char *)wbFile_read(
      file, width * channels * sizeof(unsigned char), height);

  img = wbImage_new(width, height, channels);

  imgData = wbImage_getData(img);

  charIter  = charData;
  floatIter = imgData;

  scale = 1.0f / ((float)depth);

  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
      for (kk = 0; kk < channels; kk++) {
        *floatIter = ((float)*charIter) * scale;
        floatIter++;
        charIter++;
      }
    }
  }

#ifdef LAZY_FILE_LOAD
  wbDelete(charData);
#endif

cleanup:
  wbFile_close(file);
  return img;
}

void wbPPM_export(const char *filename, wbImage_t img) {
  int ii;
  int jj;
  int kk;
  int depth;
  int width;
  int height;
  int channels;
  wbFile_t file;
  float *floatIter;
  unsigned char *charData;
  unsigned char *charIter;

  file = wbFile_open(filename, "wb+");

  width    = wbImage_getWidth(img);
  height   = wbImage_getHeight(img);
  channels = wbImage_getChannels(img);
  depth    = 255;

  if (channels == 1) {
    wbFile_writeLine(file, "P5");
  } else {
    wbFile_writeLine(file, "P6");
  }
  wbFile_writeLine(file, "#Created via wbPPM Export");
  wbFile_writeLine(file, wbString(width, " ", height));
  wbFile_writeLine(file, wbString(depth));

  charData = wbNewArray(unsigned char, width *height *channels);

  charIter  = charData;
  floatIter = wbImage_getData(img);

  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
      for (kk = 0; kk < channels; kk++) {
        *charIter = (unsigned char)ceil(_clamp(*floatIter, 0, 1) * depth);
        floatIter++;
        charIter++;
      }
    }
  }

  wbFile_write(file, charData, width * channels * sizeof(unsigned char),
               height);

  wbDelete(charData);
  wbFile_delete(file);

  return;
}
