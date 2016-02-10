

#ifndef __WB_MEMORY_MANAGER_H__
#define __WB_MEMORY_MANAGER_H__

#ifdef WB_USE_CUSTOM_MALLOC

typedef unsigned char byte;
typedef unsigned long ulong;

void memmgr_free(void *ap);

void memmgr_init(size_t heapsize);

void wbMemoryManager_new(size_t heapsize);

void *memmgr_alloc(ulong nbytes, int *err);

void memmgr_free(void *ap);

ulong memmgr_get_block_size(void *ap);

void memmgr_print_stats();

#endif /* WB_USE_CUSTOM_MALLOC */

#endif /* __WB_MEMORY_MANAGER_H__ */
