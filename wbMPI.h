
#ifndef __WB_MPI_H__
#define __WB_MPI_H__

#ifdef WB_USE_MPI

#include <string>
#include <cstring>
#include <mpi/mpi.h>

#define mpiRank wbMPI_getRank()
#define isMasterQ ((mpiRank) == 0)

extern int wbMPI_getRank();

extern int rankCount();

extern const char *wbMPI_getStringFromRank(int rank, int tag);
extern void wbMPI_sendStringToMaster(const char *str, int tag);

extern int wbMPI_Init();

extern bool finalizedQ;

extern "C" int wbMPI_Finalize(void);
extern "C" void wbMPI_Exit(void);

#define MPI_Finalize wbMPI_Finalize

#else /* WB_USE_MPI */
static inline int rankCount() {
	return 1;
}
#endif /* WB_USE_MPI */
#endif /* __WB_MPI_H__ */
