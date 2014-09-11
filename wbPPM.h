

#ifndef __wbPPM_H__
#define __wbPPM_H__

START_EXTERN_C
wbImage_t wbPPM_import(const char *filename);
void wbPPM_export(const char *filename, wbImage_t img);
END_EXTERN_C

#endif /* __wbPPM_H__ */
