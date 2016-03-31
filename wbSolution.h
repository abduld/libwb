

#ifndef __WB_SOLUTION_H__
#define __WB_SOLUTION_H__

typedef struct st_wbSolution_t {
  char * id;
  char * session_id;
  char *type;
  char *outputFile;
  void *data;
  int rows;
  int columns;
  int depth;
} wbSolution_t;

#define wbSolution_getId(sol) ((sol).id)
#define wbSolution_getSessionId(sol) ((sol).session_id)
#define wbSolution_getType(sol) ((sol).type)
#define wbSolution_getOutputFile(sol) ((sol).outputFile)
#define wbSolution_getData(sol) ((sol).data)
#define wbSolution_getRows(sol) ((sol).rows)
#define wbSolution_getColumns(sol) ((sol).columns)
#define wbSolution_getDepth(sol) ((sol).depth)

#define wbSolution_getHeight wbSolution_getRows
#define wbSolution_getWidth wbSolution_getColumns
#define wbSolution_getChannels wbSolution_getDepth

#define wbSolution_setId(sol, val) (wbSolution_getId(sol) = val)
#define wbSolution_setSessionId(sol, val)                                 \
  (wbSolution_getSessionId(sol) = val)
#define wbSolution_setType(sol, val) (wbSolution_getType(sol) = val)
#define wbSolution_setOutputFile(sol, val)                                \
  (wbSolution_getOutputFile(sol) = val)
#define wbSolution_setData(sol, val) (wbSolution_getData(sol) = val)
#define wbSolution_setRows(sol, val) (wbSolution_getRows(sol) = val)
#define wbSolution_setColumns(sol, val) (wbSolution_getColumns(sol) = val)
#define wbSolution_setDepth(sol, val) (wbSolution_getDepth(sol) = val)

wbBool wbSolution(char *expectedOutputFile, char *outputFile, char *type0,
                  void *data, int rows, int columns);
wbBool wbSolution(wbArg_t arg, void *data, int rows, int columns);
EXTERN_C wbBool wbSolution(wbArg_t arg, void *data, int rows);
wbBool wbSolution(wbArg_t arg, wbImage_t img);

#endif /* __WB_SOLUTION_H__ */
