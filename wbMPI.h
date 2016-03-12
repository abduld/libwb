
#ifndef __WB_MPI_H__
#define __WB_MPI_H__

#ifdef WB_USE_MPI

#include <cstring>
#include <mpi/mpi.h>
#include <string>

#define isMasterQ ((wbMPI_getRank()) == 0)

extern int wbMPI_getRank();

extern int rankCount();

extern const char *wbMPI_getStringFromRank(int rank, int tag);
extern void wbMPI_sendStringToMaster(const char *str, int tag);

extern int wbMPI_Init(int *argc, char ***argv);

extern bool finalizedQ;

extern "C" int wbMPI_Finalize(void);
extern "C" void wbMPI_Exit(void);

#define MPI_Finalize wbMPI_Finalize

#else  /* WB_USE_MPI */
static inline int rankCount() {
  return 1;
}
static inline int wbMPI_getRank() {
  return 0;
}
#endif /* WB_USE_MPI */
#endif /* __WB_MPI_H__ */
