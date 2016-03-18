#ifndef __WB_PATH_H__
#define __WB_PATH_H__

char *wbPath_join(const char *p1, const char *p2);
char *wbPath_join(const char *p1, const char *p2, const char *p3);
char *wbPath_join(const char *p1, const char *p2, const char *p3,
                  const char *p4);

char *wbPath_join(const std::string &p1, const std::string &p2);
char *wbPath_join(const std::string &p1, const std::string &p2,
                  const std::string &p3);
char *wbPath_join(const std::string &p1, const std::string &p2,
                  const std::string &p3, const std::string &p4);

template <typename T1, typename T2>
static char *wbPath_join(const T1 &p1, const T2 &p2) {
  return wbPath_join(wbString(p1), wbString(p2));
}
template <typename T1, typename T2, typename T3>
static char *wbPath_join(const T1 &p1, const T2 &p2, const T3 &p3) {
  return wbPath_join(wbString(p1), wbString(p2), wbString(p3));
}
template <typename T1, typename T2, typename T3, typename T4>
static char *wbPath_join(const T1 &p1, const T2 &p2, const T3 &p3,
                         const T4 &p4) {
  return wbPath_join(wbString(p1), wbString(p2), wbString(p3),
                     wbString(p4));
}

#endif /* __WB_PATH_H__ */
