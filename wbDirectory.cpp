#include "wb.h"

#ifdef WB_USE_UNIX
static const char dir_seperator = '/';
static void mkdir_(const char *dir) {
  mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
#else  /* WB_USE_LINUX */
static const char dir_seperator = '\\';
static void mkdir_(const char *dir) {
  _mkdir(dir);
}
#endif /* WB_USE_LINUX */

EXTERN_C void CreateDirectory(const char *dir) {
  char tmp[PATH_MAX];
  char *p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp), "%s", dir);
  len = strlen(tmp);
  if (tmp[len - 1] == dir_seperator)
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++)
    if (*p == dir_seperator) {
      *p = 0;
      mkdir_(tmp);
      *p = dir_seperator;
    }
  mkdir_(tmp);
}

EXTERN_C char *DirectoryName(const char *pth0) {
  char *pth = wbString_duplicate(pth0);
  char *p   = strrchr(pth, dir_seperator);
  if (p) {
    p[0] = 0;
  }
  return pth;
}
