
#include "wb.h"

char *solutionJSON = NULL;
static string _solution_correctQ("");

static void _onUnsameImageFunction(string str) {
  _solution_correctQ = str;
}

template <typename T>
static wbBool wbSolution_listCorrectQ(const char *expectedOutputFile,
                                      wbSolution_t sol, const char *type) {
  wbBool res;
  T *expectedData;
  int expectedRows, expectedColumns;

  expectedData = (T *)wbImport(expectedOutputFile, &expectedRows,
                               &expectedColumns, type);

  if (expectedData == NULL) {
    _solution_correctQ = "Failed to open expected output file.";
    res                = wbFalse;
  } else if (expectedRows != wbSolution_getRows(sol)) {
    wbLog(TRACE, "Number of rows in the solution is ",
          wbSolution_getRows(sol), ". Expected number of rows is ",
          expectedRows, ".");
    _solution_correctQ =
        "The number of rows in the solution did not match "
        "that of the expected results.";
    res = wbFalse;
  } else if (expectedColumns != wbSolution_getColumns(sol)) {
    wbLog(TRACE, "Number of columns in the solution is ",
          wbSolution_getColumns(sol), ". Expected number of columns is ",
          expectedColumns, ".");
    _solution_correctQ = "The number of columns in the solution did not "
                         "match that of the expected results.";
    res = wbFalse;
  } else {
    int ii, jj, idx;
    T *solutionData;

    solutionData = (T *)wbSolution_getData(sol);

    for (ii = 0; ii < expectedRows; ii++) {
      for (jj = 0; jj < expectedColumns; jj++) {
        idx = ii * expectedColumns + jj;
        if (wbUnequalQ(expectedData[idx], solutionData[idx])) {
          string str;
          if (expectedColumns == 1) {
            str = wbString(
                "The solution did not match the expected results at row ",
                ii, ". Expecting ", expectedData[idx], " but got ",
                solutionData[idx], ".");
          } else {
            str = wbString("The solution did not match the expected "
                           "results at column ",
                           jj, " and row ", ii, ". Expecting ",
                           expectedData[idx], " but got ",
                           solutionData[idx], ".");
          }
          _solution_correctQ = str;
          res                = wbFalse;
          goto matrixCleanup;
        }
      }
    }

    res = wbTrue;
  matrixCleanup:
    if (expectedData != NULL) {
      wbFree(expectedData);
    }
  }
  return res;
}

static wbBool wbSolution_correctQ(char *expectedOutputFile,
                                  wbSolution_t sol) {
  if (expectedOutputFile == NULL) {
    _solution_correctQ = "Failed to determined the expected output file.";
    return wbFalse;
  } else if (!wbFile_existsQ(expectedOutputFile)) {
    _solution_correctQ =
        wbString("The file ", expectedOutputFile, " does not exist.");
    return wbFalse;
  } else if (wbString_sameQ(wbSolution_getType(sol), "image")) {
    wbBool res;
    wbImage_t solutionImage = NULL;
    wbImage_t expectedImage = wbImport(expectedOutputFile);
    if (expectedImage == NULL) {
      _solution_correctQ = "Failed to open expected output file.";
      res                = wbFalse;
    } else if (wbImage_getWidth(expectedImage) !=
               wbSolution_getWidth(sol)) {
      _solution_correctQ =
          "The image width of the expected image does not "
          "match that of the solution.";
      res = wbFalse;
    } else if (wbImage_getHeight(expectedImage) !=
               wbSolution_getHeight(sol)) {
      _solution_correctQ =
          "The image height of the expected image does not "
          "match that of the solution.";
      res = wbFalse;
    } else if (wbImage_getChannels(expectedImage) !=
               wbSolution_getChannels(sol)) {
      _solution_correctQ =
          "The image channels of the expected image does not "
          "match that of the solution.";
      res = wbFalse;
    } else {
      solutionImage = (wbImage_t)wbSolution_getData(sol);
      wbAssert(solutionImage != NULL);
      res = wbImage_sameQ(solutionImage, expectedImage,
                          _onUnsameImageFunction);
    }
    if (expectedImage != NULL) {
      wbImage_delete(expectedImage);
    }
    return res;
  } else if (wbString_sameQ(wbSolution_getType(sol), "histogram")) {
    return wbSolution_listCorrectQ<unsigned char>(expectedOutputFile, sol,
                                                  "Integer");
  } else if (wbString_sameQ(wbSolution_getType(sol), "integral_vector")) {
    return wbSolution_listCorrectQ<int>(expectedOutputFile, sol,
                                        "Integer");
  } else if (wbString_sameQ(wbSolution_getType(sol), "vector") ||
             wbString_sameQ(wbSolution_getType(sol), "matrix")) {
    return wbSolution_listCorrectQ<wbReal_t>(expectedOutputFile, sol,
                                             "Real");
  } else {
    wbAssert(wbFalse);
    return wbFalse;
  }
}

wbBool wbSolution(char *expectedOutputFile, char *outputFile, char *type0,
                  void *data, int rows, int columns, int depth) {
  char *type;
  wbBool res;
  wbSolution_t sol;

  if (expectedOutputFile == NULL || data == NULL || type0 == NULL) {
    wbLog(ERROR, "Failed to grade solution");
    return wbFalse;
  }

  type = wbString_toLower(type0);

  if (_solution_correctQ != "") {
    _solution_correctQ = "";
  }

  wbSolution_setOutputFile(sol, outputFile);
  wbSolution_setId(sol, uuid());
  wbSolution_setSessionId(sol, sessionId());
  wbSolution_setType(sol, type);
  wbSolution_setData(sol, data);
  wbSolution_setRows(sol, rows);
  wbSolution_setColumns(sol, columns);
  wbSolution_setDepth(sol, depth);

  res = wbSolution_correctQ(expectedOutputFile, sol);

  if (outputFile != NULL) {
    if (wbString_sameQ(type, "image")) {
      wbImage_t inputImage = (wbImage_t)data;
      wbImage_t img        = wbImage_new(wbImage_getWidth(inputImage),
                                  wbImage_getHeight(inputImage),
                                  wbImage_getChannels(inputImage));
      memcpy(wbImage_getData(img), wbImage_getData(inputImage),
             rows * columns * wbImage_channels * sizeof(wbReal_t));
      wbExport(outputFile, img);
      wbImage_delete(img);
    } else if (wbString_sameQ(type, "integral_vector")) {
      wbExport(outputFile, (int *)data, rows, columns);
    } else if (wbString_sameQ(type, "vector") ||
               wbString_sameQ(type, "matrix")) {
      wbExport(outputFile, (wbReal_t *)data, rows, columns);
    } else if (wbString_sameQ(type, "histogram")) {
      wbExport(outputFile, (unsigned char *)data, rows, columns);
    } else if (wbString_sameQ(type, "text")) {
      wbExport_text(outputFile, (unsigned char *)data, rows * columns);
    }
  }

  wbFree(type);

  return res;
}

wbBool wbSolution(char *expectedOutputFile, char *outputFile, char *type0,
                  void *data, int rows, int columns) {
  return wbSolution(expectedOutputFile, outputFile, type0, data, rows,
                    columns, 1);
}

wbBool wbSolution(wbArg_t arg, void *data, int rows, int columns,
                  int depth) {
  char *type;
  wbBool res;
  char *expectedOutputFile;
  char *outputFile;
  stringstream ss;

  expectedOutputFile = wbArg_getExpectedOutputFile(arg);
  outputFile         = wbArg_getOutputFile(arg);
  type               = wbArg_getType(arg);

  wbAssert(type != NULL);
  wbAssert(expectedOutputFile != NULL);
  wbAssert(outputFile != NULL);

  res = wbSolution(expectedOutputFile, outputFile, type, data, rows,
                   columns, depth);

  if (WB_USE_JSON11) {
    json11::Json json;
    if (res) {
      json = json11::Json::object{{"correctq", true},
                                  {"message", "The solution is correct"}};
    } else {
      json = json11::Json::object{{"correctq", false},
                                  {"message", _solution_correctQ}};
    }

#ifdef wbLogger_printOnLog
    if (wbLogger_printOnLog) {
      json11::Json e =
          json11::Json::object{{"type", "solution"}, {"data", json}};
      std::cout << e.dump() << std::endl;
    }
#endif /* wbLogger_printOnLog */

    solutionJSON = wbString_duplicate(json.string_value());
  } else {
    if (res) {
      ss << "{\n";
      ss << wbString_quote("correctq") << ": true,\n";
      ss << wbString_quote("message") << ": "
         << wbString_quote("Solution is correct.") << "\n";
      ss << "}";
    } else {
      ss << "{\n";
      ss << wbString_quote("correctq") << ": false,\n";
      ss << wbString_quote("message") << ": "
         << wbString_quote(_solution_correctQ) << "\n";
      ss << "}";
    }
    solutionJSON = wbString_duplicate(ss.str());
  }

  return res;
}

wbBool wbSolution(wbArg_t arg, void *data, int rows, int columns) {
  return wbSolution(arg, data, rows, columns, 1);
}

EXTERN_C wbBool wbSolution(wbArg_t arg, void *data, int rows) {
  return wbSolution(arg, data, rows, 1);
}

wbBool wbSolution(wbArg_t arg, wbImage_t img) {
  return wbSolution(arg, img, wbImage_getHeight(img),
                    wbImage_getWidth(img), wbImage_getChannels(img));
}
