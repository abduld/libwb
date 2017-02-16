
#ifndef __WB_SORT_H__
#define __WB_SORT_H__


template <typename T>
static void wbSort(T *data, int start, int end) {
  if ((end - start + 1) > 1) {
    int left = start, right = end;
    int pivot = data[right];
    while (left <= right) {
      while (data[left] > pivot) {
        left = left + 1;
      }
      while (data[right] < pivot) {
        right = right - 1;
      }
      if (left <= right) {
        int tmp     = data[left];
        data[left]  = data[right];
        data[right] = tmp;
        left        = left + 1;
        right       = right - 1;
      }
    }
    wbSort(data, start, right);
    wbSort(data, left, end);
  }
}

template <typename T>
static void wbSort(T *data, int len)  {
    return wbSort<T>(data, 0, len);
}

template <typename T, typename K>
static void wbSortByKey(T *data, K *key, int start, int end) {
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
    wbSortByKey(data, key, start, right);
    wbSortByKey(data, key, left, end);
  }
}

template <typename T, typename K>
static void wbSortByKey(T *data, K *key, int len) {
    return wbSortByKey<T, K>(data, key, len);
}


#endif // __WB_SORT_H__
