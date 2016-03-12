
#include "wb.h"

static void sort(int *data, int *key, int start, int end) {
  if ((end - start + 1) > 1) {
    int left = start, right = end;
    int pivot = key[right];
    while (left <= right) {
      while (key[left] > pivot) {
        left = left + 1;
      }
      while (key[right] < pivot) {
        right = right - 1;
      }
      if (left <= right) {
        int tmp     = key[left];
        key[left]   = key[right];
        key[right]  = tmp;
        tmp         = data[left];
        data[left]  = data[right];
        data[right] = tmp;
        left        = left + 1;
        right       = right - 1;
      }
    }
    sort(data, key, start, right);
    sort(data, key, left, end);
  }
}

EXTERN_C void CSRToJDS(int dim, int *csrRowPtr, int *csrColIdx,
                       float *csrData, int **jdsRowPerm, int **jdsRowNNZ,
                       int **jdsColStartIdx, int **jdsColIdx,
                       float **jdsData) {
  // Row Permutation Vector
  *jdsRowPerm = (int *)malloc(sizeof(int) * dim);
  for (int rowIdx = 0; rowIdx < dim; ++rowIdx) {
    (*jdsRowPerm)[rowIdx] = rowIdx;
  }

  // Number of non-zeros per row
  *jdsRowNNZ = (int *)malloc(sizeof(int) * dim);
  for (int rowIdx = 0; rowIdx < dim; ++rowIdx) {
    (*jdsRowNNZ)[rowIdx] = csrRowPtr[rowIdx + 1] - csrRowPtr[rowIdx];
  }

  // Sort rows by number of non-zeros
  sort(*jdsRowPerm, *jdsRowNNZ, 0, dim - 1);

  // Starting point of each compressed column
  int maxRowNNZ = (*jdsRowNNZ)[0]; // Largest number of non-zeros per row
  DEBUG(printf("jdsRowNNZ = %d\n", maxRowNNZ));
  *jdsColStartIdx      = (int *)malloc(sizeof(int) * maxRowNNZ);
  (*jdsColStartIdx)[0] = 0; // First column starts at 0
  for (int col = 0; col < maxRowNNZ - 1; ++col) {
    // Count the number of rows with entries in this column
    int count = 0;
    for (int idx = 0; idx < dim; ++idx) {
      if ((*jdsRowNNZ)[idx] > col) {
        ++count;
      }
    }
    (*jdsColStartIdx)[col + 1] = (*jdsColStartIdx)[col] + count;
  }

  // Sort the column indexes and data
  const int NNZ = csrRowPtr[dim];
  DEBUG(printf("NNZ = %d\n", NNZ));
  *jdsColIdx = (int *)malloc(sizeof(int) * NNZ);
  DEBUG(printf("dim = %d\n", dim));
  *jdsData = (float *)malloc(sizeof(float) * NNZ);
  for (int idx = 0; idx < dim; ++idx) { // For every row
    int row    = (*jdsRowPerm)[idx];
    int rowNNZ = (*jdsRowNNZ)[idx];
    for (int nnzIdx = 0; nnzIdx < rowNNZ; ++nnzIdx) {
      int jdsPos           = (*jdsColStartIdx)[nnzIdx] + idx;
      int csrPos           = csrRowPtr[row] + nnzIdx;
      (*jdsColIdx)[jdsPos] = csrColIdx[csrPos];
      (*jdsData)[jdsPos]   = csrData[csrPos];
    }
  }
}
