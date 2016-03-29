#ifndef __WB_DIRECTORY__
#define __WB_DIRECTORY__

extern const char wbDirectorySeperator;
EXTERN_C char *wbDirectory_name(const char *pth);
EXTERN_C const char *wbDirectory_create(const char *dir);
EXTERN_C char *wbDirectory_current();

#endif /* __WB_DIRECTORY__ */
