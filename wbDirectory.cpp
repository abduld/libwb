#include "wb.h"

#ifndef PATH_MAX
#ifdef FILENAME_MAX
#define PATH_MAX FILENAME_MAX
#else /* FILENAME_MAX */
#define PATH_MAX 4096
#endif /* FILENAME_MAX */
#endif /* PATH_MAX */

#ifdef WB_USE_UNIX
const char wbDirectorySeperator = '/';
static char *getcwd_(char *buf, int maxLen) {
  return getcwd(buf, maxLen);
}
static void mkdir_(const char *dir) {
  mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
#else  /* WB_USE_LINUX */
const char wbDirectorySeperator = '\\';
static char *getcwd_(char *buf, int maxLen) {
  return _getcwd(buf, maxLen);
}
static void mkdir_(const char *dir) {
  _mkdir(dir);
}
#endif /* WB_USE_LINUX */

EXTERN_C const char *wbDirectory_create(const char *dir) {
  char tmp[PATH_MAX];
  char *p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp), "%s", dir);
  len = strlen(tmp);
  if (tmp[len - 1] == wbDirectorySeperator) {
    tmp[len - 1] = 0;
  }
  for (p = tmp + 1; *p; p++) {
    if (*p == wbDirectorySeperator) {
      *p = 0;
      mkdir_(tmp);
      *p = wbDirectorySeperator;
    }
  }
  mkdir_(tmp);
  return dir;
}

EXTERN_C char *wbDirectory_name(const char *pth0) {
  char *pth = wbString_duplicate(pth0);
  char *p   = strrchr(pth, wbDirectorySeperator);
  if (p) {
    p[0] = 0;
  }
  return pth;
}

EXTERN_C char *wbDirectory_current() {
  char *tmp = wbNewArray(char, PATH_MAX + 1);
  if (getcwd_(tmp, PATH_MAX)) {
    return tmp;
  }

  wbDelete(tmp);

  int error = errno;
  switch (error) {
    case EACCES:
      std::cerr
          << "Cannot get current directory :: access denied. exiting..."
          << std::endl;
      exit(-1);
    case ENOMEM:
      std::cerr << "Cannot get current directory :: insufficient storage. "
                   "exiting..."
                << std::endl;
      exit(-1);
    default:
      std::cerr << "Cannot get current directory :: unrecognised error "
                << error << std::endl;
      exit(-1);
  }
}