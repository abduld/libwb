
#ifdef WB_USE_CUSTOM_MALLOC

#include <wb.h>
#include <sys/mman.h>

//
// Memory manager: dynamically allocates memory from
// a fixed pool that is allocated statically at link-time.
//
// Usage: after calling memmgr_init() in your
// initialization routine, just use memmgr_alloc() instead
// of malloc() and memmgr_free() instead of free().
// Naturally, you can use the preprocessor to define
// malloc() and free() as aliases to memmgr_alloc() and
// memmgr_free(). This way the manager will be a drop-in
// replacement for the standard C library allocators, and can
// be useful for debugging memory allocation problems and
// leaks.
//
// Preprocessor flags you can define to customize the
// memory manager:
//
// DEBUG_MEMMGR_FATAL
//    Allow printing out a message when allocations fail
//
// DEBUG_MEMMGR_SUPPORT_STATS
//    Allow printing out of stats in function
//    memmgr_print_stats When this is disabled,
//    memmgr_print_stats does nothing.
//
// Note that in production code on an embedded system
// you'll probably want to keep those undefined, because
// they cause printf to be called.
//
// MIN_POOL_ALLOC_QUANTAS
//    Internally, the memory manager allocates memory in
//    quantas roughly the size of two ulong objects. To
//    minimize pool fragmentation in case of multiple allocations
//    and deallocations, it is advisable to not allocate
//    blocks that are too small.
//    This flag sets the minimal ammount of quantas for
//    an allocation. If the size of a ulong is 4 and you
//    set this flag to 16, the minimal size of an allocation
//    will be 4 * 2 * 16 = 128 bytes
//    If you have a lot of small allocations, keep this value
//    low to conserve memory. If you have mostly large
//    allocations, it is best to make it higher, to avoid
//    fragmentation.
//
// Notes:
// 1. This memory manager is *not thread safe*. Use it only
//    for single thread/task applications.
//

#define DEBUG_MEMMGR_SUPPORT_STATS 1

#define MIN_POOL_ALLOC_QUANTAS 128

typedef ulong Align;

union mem_header_union {
  struct {
    // Pointer to the next block in the free list
    //
    union mem_header_union *next;

    // Size of the block (in quantas of sizeof(mem_header_t))
    //
    ulong size;
  } s;

  // Used to align headers in memory to a boundary
  //
  Align align_dummy;
};

typedef union mem_header_union mem_header_t;

// Initial empty list
//
static mem_header_t base;

// Start of free list
//
static mem_header_t *freep = 0;

// Static pool for new allocations
//
static byte *pool;
static size_t pool_size;
static ulong pool_free_pos = 0;

void memmgr_free(void *ap);

void memmgr_init(size_t heapsize) {
  void *heap = mmap(0, (size_t)heapsize, PROT_READ | PROT_WRITE,
#ifndef __APPLE__
                    MAP_ANONYMOUS |
#endif /* __APPLE__ */
                        MAP_SHARED,
                    -1, 0);
  if (heap == MAP_FAILED) {
    // Couldn't allocate heap memory.
    exit(22);
  }
  pool = (byte *)heap;
  pool_size = heapsize;

  base.s.next = 0;
  base.s.size = 0;
  freep = 0;
  pool_free_pos = 0;
}

void wbMemoryManager_new(size_t heapsize) {
  memmgr_init(heapsize);
  return;
}

void memmgr_print_stats() {
  mem_header_t *p;

  printf("------ Memory manager stats ------\n\n");
  printf("Pool: free_pos = %lu (%lu bytes left)\n\n", pool_free_pos,
         pool_size - pool_free_pos);

  p = (mem_header_t *)pool;

  while (p < (mem_header_t *)(pool + pool_free_pos)) {
    printf("  * Addr: 0x%8lu; Size: %8lu\n", (ulong)p, p->s.size);

    p += p->s.size;
  }

  printf("\nFree list:\n\n");

  if (freep) {
    p = freep;

    while (1) {
      printf("  * Addr: 0x%8lu; Size: %8lu; Next: 0x%8lu\n", (ulong)p,
             p->s.size, (ulong)p->s.next);

      p = p->s.next;

      if (p == freep)
        break;
    }
  } else {
    printf("Empty\n");
  }

  printf("\n");
}

static mem_header_t *get_mem_from_pool(ulong nquantas) {
  ulong total_req_size;

  mem_header_t *h;

  if (nquantas < MIN_POOL_ALLOC_QUANTAS)
    nquantas = MIN_POOL_ALLOC_QUANTAS;

  total_req_size = nquantas * sizeof(mem_header_t);

  if (pool_free_pos + total_req_size <= pool_size) {
    h = (mem_header_t *)(pool + pool_free_pos);
    h->s.size = nquantas;
    memmgr_free((void *)(h + 1));
    pool_free_pos += total_req_size;
  } else {
    return 0;
  }

  return freep;
}

// Allocations are done in 'quantas' of header size.
// The search for a free block of adequate size begins at the point 'freep'
// where the last block was found.
// If a too-big block is found, it is split and the tail is returned (this
// way the header of the original needs only to have its size adjusted).
// The pointer returned to the user points to the free space within the block,
// which begins one quanta after the header.
//
void *memmgr_alloc(ulong nbytes, int *err) {
  mem_header_t *p;
  mem_header_t *prevp;

  *err = 0;

  // Calculate how many quantas are required: we need enough to house all
  // the requested bytes, plus the header. The -1 and +1 are there to make sure
  // that if nbytes is a multiple of nquantas, we don't allocate too much
  //
  ulong nquantas =
      (nbytes + sizeof(mem_header_t) - 1) / sizeof(mem_header_t) + 1;

  // First alloc call, and no free list yet ? Use 'base' for an initial
  // denegerate block of size 0, which points to itself
  //
  if ((prevp = freep) == 0) {
    base.s.next = freep = prevp = &base;
    base.s.size = 0;
  }

  for (p = prevp->s.next;; prevp = p, p = p->s.next) {
    // big enough ?
    if (p->s.size >= nquantas) {
      // exactly ?
      if (p->s.size == nquantas) {
        // just eliminate this block from the free list by pointing
        // its prev's next to its next
        //
        prevp->s.next = p->s.next;
      } else // too big
      {
        p->s.size -= nquantas;
        p += p->s.size;
        p->s.size = nquantas;
      }

      freep = prevp;
      return (void *)(p + 1);
    }
    // Reached end of free list ?
    // Try to allocate the block from the pool. If that succeeds,
    // get_mem_from_pool adds the new block to the free list and
    // it will be found in the following iterations. If the call
    // to get_mem_from_pool doesn't succeed, we've run out of
    // memory
    //
    else if (p == freep) {
      if ((p = get_mem_from_pool(nquantas)) == 0) {
        *err = 1;
        return 0;
      }
    }
  }
}

// Scans the free list, starting at freep, looking the the place to insert the
// free block. This is either between two existing blocks or at the end of the
// list. In any case, if the block being freed is adjacent to either neighbor,
// the adjacent blocks are combined.
//
void memmgr_free(void *ap) {
  mem_header_t *block;
  mem_header_t *p;

  // acquire pointer to block header
  block = ((mem_header_t *)ap) - 1;

  // Find the correct place to place the block in (the free list is sorted by
  // address, increasing order)
  //
  for (p = freep; p && !(block > p && block < p->s.next); p = p->s.next) {
    // Since the free list is circular, there is one link where a
    // higher-addressed block points to a lower-addressed block.
    // This condition checks if the block should be actually
    // inserted between them
    //
    if (p >= p->s.next && (block > p || block < p->s.next))
      break;
  }

  // Try to combine with the higher neighbor
  //
  if (block + block->s.size == p->s.next) {
    block->s.size += p->s.next->s.size;
    block->s.next = p->s.next->s.next;
  } else {
    block->s.next = p->s.next;
  }

  // Try to combine with the lower neighbor
  //
  if (p + p->s.size == block) {
    p->s.size += block->s.size;
    p->s.next = block->s.next;
  } else {
    p->s.next = block;
  }

  freep = p;
}

// Find out the allocation size of given block.
// Needed to implement realloc() and similar functions.
ulong memmgr_get_block_size(void *ap) {
  mem_header_t *block = ((mem_header_t *)ap) - 1;
  return block->s.size;
}

#endif /* WB_USE_CUSTOM_MALLOC */
