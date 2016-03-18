#include "wb.h"

char *wbPath_join(const char *p1, const char *p2) {
  size_t s1 = strlen(p1);
  size_t s2 = strlen(p2);
  char *res =
      wbNewArray(char, s1 + 1 /* seperator */ + s2 + 1 /* terminator */);
  memcpy(res, p1, s1);
  char *iter = res + s1;
  *iter++    = wbDirectorySeperator;
  memcpy(iter, p2, s2);
  iter += s2;
  *iter = '\0';
  return res;
}

char *wbPath_join(const char *p1, const char *p2, const char *p3) {
  char *p12 = wbPath_join(p1, p2);
  char *res = wbPath_join(p12, p3);
  wbDelete(p12);
  return res;
}

char *wbPath_join(const char *p1, const char *p2, const char *p3,
                  const char *p4) {
  char *p123 = wbPath_join(p1, p2, p3);
  char *res  = wbPath_join(p123, p4);
  wbDelete(p123);
  return res;
}

char *wbPath_join(const std::string &p1, const std::string &p2) {
  return wbPath_join(p1.c_str(), p2.c_str());
}
char *wbPath_join(const std::string &p1, const std::string &p2,
                  const std::string &p3) {
  return wbPath_join(p1.c_str(), p2.c_str(), p3.c_str());
}
char *wbPath_join(const std::string &p1, const std::string &p2,
                  const std::string &p3, const std::string &p4) {
  return wbPath_join(p1.c_str(), p2.c_str(), p3.c_str(), p4.c_str());
}