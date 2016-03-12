#ifndef __WB_UTILS_H__
#define __WB_UTILS_H__

#ifdef WB_DEBUG
#define DEBUG(...) __VA_ARGS__
#else /* WB_DEBUG */
#define DEBUG(...)
#endif /* WB_DEBUG */

#endif /* __WB_UTILS_H__ */
