

#ifndef __WB_INIT_H__
#define __WB_INIT_H__

#ifndef _MSC_VER
__attribute__((__constructor__))
#endif /* _MSC_VER */
void wb_init(int *argc, char ***argv);

#endif /* __WB_INIT_H__ */
