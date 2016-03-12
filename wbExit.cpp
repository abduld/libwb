
#include <wb.h>

enum {
  wbMPI_timerTag          = 2,
  wbMPI_loggerTag         = 4,
  wbMPI_solutionExistsTag = 8,
  wbMPI_solutionTag       = 16
};

void wb_atExit(void) {
  using std::cout;
  using std::endl;

#ifdef WB_USE_CUDA
  cudaDeviceSynchronize();
#endif /* WB_USE_CUDA */

  int nranks = rankCount();
  if (nranks > 1) {
#ifdef WB_USE_MPI
    if (isMasterQ) {
      cout << "==$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;

      cout << "{\n";
      cout << wbString_quote("timer") << ":";
      cout << "[\n";
      for (int ii = 0; ii < nranks; ii++) {
        if (ii == 0) {
          cout << wbTimer_toJSON();
        } else {
          const char *msg = wbMPI_getStringFromRank(ii, wbMPI_timerTag);
          if (msg != NULL && strlen(msg) != 0) {
            cout << ",\n";
            cout << msg;
            //		 free(msg);
          }
        }
      }
      cout << "]" << endl; // close timer

      cout << "," << endl; // start logger
      cout << wbString_quote("logger") << ":";
      cout << "[\n";
      for (int ii = 0; ii < nranks; ii++) {
        if (ii == 0) {
          cout << wbLogger_toJSON();
        } else {
          const char *msg = wbMPI_getStringFromRank(ii, wbMPI_loggerTag);
          if (msg != NULL && strlen(msg) != 0) {
            cout << ",\n";
            cout << msg;
            //		 free(msg);
          }
        }
      }
      cout << "]" << endl; // close logger

      cout << "," << endl; // start solutionExists
      cout << wbString_quote("cuda_memory") << ":" << _cudaMallocSize
           << ",\n";

      if (solutionJSON) {
        cout << wbString_quote("solution_exists") << ": true,\n";
        cout << wbString_quote("solution") << ":" << solutionJSON << "\n";
      } else {
        cout << wbString_quote("solution_exists") << ": false\n";
      }
      cout << "}" << endl; // close json

    } else {
      wbMPI_sendStringToMaster(wbTimer_toJSON().c_str(), wbMPI_timerTag);
      wbMPI_sendStringToMaster(wbLogger_toJSON().c_str(), wbMPI_loggerTag);
    }

#endif /* WB_USE_MPI */
  } else {
    cout << "==$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;

    cout << "{\n" << wbString_quote("timer") << ":[" << wbTimer_toJSON()
         << "],\n" << wbString_quote("logger") << ":[" << wbLogger_toJSON()
         << "],\n";

#ifdef WB_USE_CUDA
    cout << wbString_quote("cuda_memory") << ":" << _cudaMallocSize
         << ",\n";
#endif /* WB_USE_CUDA */

    if (solutionJSON) {
      cout << wbString_quote("solution_exists") << ": true,\n";
      cout << wbString_quote("solution") << ":" << solutionJSON << "\n";
    } else {
      cout << wbString_quote("solution_exists") << ": false\n";
    }
    cout << "}" << endl;
  }

  // wbTimer_delete(_timer);
  // wbLogger_delete(_logger);

  _timer  = NULL;
  _logger = NULL;

// wbFile_atExit();

#ifdef WB_DEBUG_MEMMGR_SUPPORT_STATS
  memmgr_print_stats();
#endif /* WB_DEBUG_MEMMGR_SUPPORT_STATS */

#ifdef WB_USE_CUDA
  cudaDeviceReset();
#endif

  exit(0);

  // assert(0);

  return;
}
