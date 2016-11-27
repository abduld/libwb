#include "wb.h"

#ifdef WB_USE_WINDOWS
uint64_t _hrtime_frequency = 0;
#endif /* WB_USE_WINDOWS */
wbTimer_t _timer = nullptr;

#ifdef WB_USE_DARWIN
static double o_timebase    = 0;
static uint64_t o_timestart = 0;
#endif /* WB_USE_DARWIN */

uint64_t _hrtime(void) {
#define NANOSEC ((uint64_t)1e9)
#ifdef WB_USE_WINDOWS
  LARGE_INTEGER counter;
  if (!QueryPerformanceCounter(&counter)) {
    return 0;
  }
  return ((uint64_t)counter.LowPart * NANOSEC / _hrtime_frequency) +
         (((uint64_t)counter.HighPart * NANOSEC / _hrtime_frequency)
          << 32);
#else
  struct timespec ts;
#ifdef WB_USE_DARWIN
#define O_NANOSEC (+1.0E-9)
#define O_GIGA UINT64_C(1000000000)
  if (!o_timestart) {
    mach_timebase_info_data_t tb{};
    mach_timebase_info(&tb);
    o_timebase = tb.numer;
    o_timebase /= tb.denom;
    o_timestart = mach_absolute_time();
  }
  double diff = (mach_absolute_time() - o_timestart) * o_timebase;
  ts.tv_sec   = diff * O_NANOSEC;
  ts.tv_nsec  = diff - (ts.tv_sec * O_GIGA);
#undef O_NANOSEC
#undef O_GIGA
#else  /* WB_USE_DARWIN */
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif /* WB_USE_DARWIN */
  return (((uint64_t)ts.tv_sec) * NANOSEC + ts.tv_nsec);
#endif /* WB_USE_WINDOWS */
#undef NANOSEC
}

static inline uint64_t getTime(void) {
#ifdef WB_USE_CUDA
  cudaDeviceSynchronize();
#endif /* WB_USE_CUDA */
  return _hrtime();
}

static inline wbTimerNode_t wbTimerNode_new(int idx, wbTimerKind_t kind,
                                            const char *file,
                                            const char *fun,
                                            int startLine) {
  wbTimerNode_t node = wbNew(struct st_wbTimerNode_t);
  wbTimerNode_setId(node, uuid());
  wbTimerNode_setIdx(node, idx);
  wbTimerNode_setSessionId(node, sessionId());
  wbTimerNode_setMPIRank(node, wbMPI_getRank());
  wbTimerNode_setLevel(node, 0);
  wbTimerNode_setStoppedQ(node, wbFalse);
  wbTimerNode_setKind(node, kind);
  wbTimerNode_setStartTime(node, 0);
  wbTimerNode_setEndTime(node, 0);
  wbTimerNode_setElapsedTime(node, 0);
  wbTimerNode_setStartLine(node, startLine);
  wbTimerNode_setEndLine(node, 0);
  wbTimerNode_setStartFunction(node, fun);
  wbTimerNode_setEndFunction(node, NULL);
  wbTimerNode_setStartFile(node, file);
  wbTimerNode_setEndFile(node, NULL);
  wbTimerNode_setNext(node, NULL);
  wbTimerNode_setPrevious(node, NULL);
  wbTimerNode_setParent(node, NULL);
  wbTimerNode_setMessage(node, NULL);
  return node;
}

static inline void wbTimerNode_delete(wbTimerNode_t node) {
  if (node != nullptr) {
    if (wbTimerNode_getMessage(node)) {
      wbDelete(wbTimerNode_getMessage(node));
    }
    wbDelete(node);
  }
}

static inline const char *_nodeKind(wbTimerKind_t kind) {
  switch (kind) {
    case wbTimerKind_Generic:
      return "Generic";
    case wbTimerKind_IO:
      return "IO";
    case wbTimerKind_GPU:
      return "GPU";
    case wbTimerKind_Copy:
      return "Copy";
    case wbTimerKind_Driver:
      return "Driver";
    case wbTimerKind_CopyAsync:
      return "CopyAsync";
    case wbTimerKind_Compute:
      return "Compute";
    case wbTimerKind_CPUGPUOverlap:
      return "CPUGPUOverlap";
  }
  return "Undefined";
}

static inline json11::Json wbTimerNode_toJSONObject(wbTimerNode_t node) {
  if (node == nullptr) {
    return json11::Json{};
  }
  int parent_id = wbTimerNode_hasParent(node)
                      ? wbTimerNode_getIdx(wbTimerNode_getParent(node))
                      : -1;
  json11::Json json = json11::Json::object{
      {"id", wbTimerNode_getId(node)},
      {"session_id", wbTimerNode_getSessionId(node)},
      {"idx", wbTimerNode_getIdx(node)},
      {"mpi_rank", wbTimerNode_getMPIRank(node)},
      {"stopped", wbTimerNode_stoppedQ(node)},
      {"kind", _nodeKind(wbTimerNode_getKind(node))},
      {"start_time", wbTimerNode_getStartTime(node)},
      {"end_time", wbTimerNode_getEndTime(node)},
      {"elapsed_time", wbTimerNode_getElapsedTime(node)},
      {"start_line", wbTimerNode_getStartLine(node)},
      {"end_line", wbTimerNode_getEndLine(node)},
      {"start_function", wbTimerNode_getStartFunction(node)},
      {"end_function", wbTimerNode_getEndFunction(node)},
      {"start_file", wbTimerNode_getStartFile(node)},
      {"end_file", wbTimerNode_getEndFile(node)},
      {"parent_id", parent_id},
      {"message", wbTimerNode_getMessage(node)},
  };
  return json;
}

static inline string wbTimerNode_toJSON(wbTimerNode_t node) {
  if (node == nullptr) {
    return "";
  } else if (WB_USE_JSON11) {
    json11::Json json = wbTimerNode_toJSONObject(node);
    return json.string_value();
  } else {
    stringstream ss;

    ss << "{\n";
    ss << wbString_quote("idx") << ":" << wbTimerNode_getIdx(node)
       << ",\n";
    ss << wbString_quote("id") << ":"
       << wbString_quote(wbTimerNode_getId(node)) << ",\n";
    ss << wbString_quote("session_id") << ":"
       << wbString_quote(wbTimerNode_getSessionId(node)) << ",\n";
    ss << wbString_quote("mpi_rank") << ":" << wbTimerNode_getMPIRank(node)
       << ",\n";
    ss << wbString_quote("stopped") << ":"
       << wbString(wbTimerNode_stoppedQ(node) ? "true" : "false") << ",\n";
    ss << wbString_quote("kind") << ":"
       << wbString_quote(_nodeKind(wbTimerNode_getKind(node))) << ",\n";
    ss << wbString_quote("start_time") << ":"
       << wbTimerNode_getStartTime(node) << ",\n";
    ss << wbString_quote("end_time") << ":" << wbTimerNode_getEndTime(node)
       << ",\n";
    ss << wbString_quote("elapsed_time") << ":"
       << wbTimerNode_getElapsedTime(node) << ",\n";
    ss << wbString_quote("start_line") << ":"
       << wbTimerNode_getStartLine(node) << ",\n";
    ss << wbString_quote("end_line") << ":" << wbTimerNode_getEndLine(node)
       << ",\n";
    ss << wbString_quote("start_function") << ":"
       << wbString_quote(wbTimerNode_getStartFunction(node)) << ",\n";
    ss << wbString_quote("end_function") << ":"
       << wbString_quote(wbTimerNode_getEndFunction(node)) << ",\n";
    ss << wbString_quote("start_file") << ":"
       << wbString_quote(wbTimerNode_getStartFile(node)) << ",\n";
    ss << wbString_quote("end_file") << ":"
       << wbString_quote(wbTimerNode_getEndFile(node)) << ",\n";
    ss << wbString_quote("parent_id") << ":"
       << wbString(wbTimerNode_hasParent(node)
                       ? wbTimerNode_getIdx(wbTimerNode_getParent(node))
                       : -1)
       << ",\n";
    ss << wbString_quote("message") << ":"
       << wbString_quote(wbTimerNode_getMessage(node)) << "\n";
    ss << "}";

    return ss.str();
  }
}

static inline string wbTimerNode_toXML(wbTimerNode_t node) {
  if (node == nullptr) {
    return "";
  } else {
    stringstream ss;

    ss << "<node>\n";
    ss << "<idx>" << wbTimerNode_getIdx(node) << "</id>\n";
    ss << "<id>" << wbTimerNode_getId(node) << "</id>\n";
    ss << "<session_id>" << wbTimerNode_getSessionId(node) << "</id>\n";
    ss << "<stoppedQ>"
       << wbString(wbTimerNode_stoppedQ(node) ? "true" : "false")
       << "</stoppedQ>\n";
    ss << "<kind>" << _nodeKind(wbTimerNode_getKind(node)) << "</kind>\n";
    ss << "<start_time>" << wbTimerNode_getStartTime(node)
       << "</start_time>\n";
    ss << "<end_time>" << wbTimerNode_getEndTime(node) << "</end_time>\n";
    ss << "<elapsed_time>" << wbTimerNode_getElapsedTime(node)
       << "</elapsed_time>\n";
    ss << "<start_line>" << wbTimerNode_getStartLine(node)
       << "</start_line>\n";
    ss << "<end_line>" << wbTimerNode_getEndLine(node) << "</end_line>\n";
    ss << "<start_function>" << wbTimerNode_getStartFunction(node)
       << "</start_function>\n";
    ss << "<end_function>" << wbTimerNode_getEndFunction(node)
       << "</end_function>\n";
    ss << "<start_file>" << wbTimerNode_getStartFile(node)
       << "</start_file>\n";
    ss << "<end_file>" << wbTimerNode_getEndFile(node) << "</end_file>\n";
    ss << "<parent_id>"
       << wbString(wbTimerNode_hasParent(node)
                       ? wbTimerNode_getIdx(wbTimerNode_getParent(node))
                       : -1)
       << "</parent_id>\n";
    ss << "<message>" << wbTimerNode_getMessage(node) << "</message>\n";
    ss << "</node>\n";

    return ss.str();
  }
}

#define wbTimer_getId(timer) ((timer)->id)
#define wbTimer_getSessionId(timer) ((timer)->session_id)
#define wbTimer_getLength(timer) ((timer)->length)
#define wbTimer_getHead(timer) ((timer)->head)
#define wbTimer_getTail(timer) ((timer)->tail)
#define wbTimer_getStartTime(timer) ((timer)->startTime)
#define wbTimer_getEndTime(timer) ((timer)->endTime)
#define wbTimer_getElapsedTime(timer) ((timer)->elapsedTime)

#define wbTimer_setId(timer, val) (wbTimer_getId(timer) = val)
#define wbTimer_setSessionId(timer, val)                                  \
  (wbTimer_getSessionId(timer) = val)
#define wbTimer_setLength(timer, val) (wbTimer_getLength(timer) = val)
#define wbTimer_setHead(timer, val) (wbTimer_getHead(timer) = val)
#define wbTimer_setTail(timer, val) (wbTimer_getTail(timer) = val)
#define wbTimer_setStartTime(node, val) (wbTimer_getStartTime(node) = val)
#define wbTimer_setEndTime(node, val) (wbTimer_getEndTime(node) = val)
#define wbTimer_setElapsedTime(node, val)                                 \
  (wbTimer_getElapsedTime(node) = val)

#define wbTimer_incrementLength(timer) (wbTimer_getLength(timer)++)
#define wbTimer_decrementLength(timer) (wbTimer_getLength(timer)--)

#define wbTimer_emptyQ(timer) (wbTimer_getLength(timer) == 0)

void wbTimer_delete(wbTimer_t timer) {
  if (timer != nullptr) {
    wbTimerNode_t tmp, iter;

    iter = wbTimer_getHead(timer);
    while (iter) {
      tmp = wbTimerNode_getNext(iter);
      wbTimerNode_delete(iter);
      iter = tmp;
    }

    wbDelete(timer);
  }
}

static json11::Json wbTimer_toJSONObject(wbTimer_t timer) {

  stringstream ss;
  wbTimerNode_t iter;
  uint64_t currentTime;
  std::vector<json11::Json> elems;

  currentTime = getTime();

  wbTimer_setEndTime(timer, currentTime);
  wbTimer_setElapsedTime(timer, currentTime - wbTimer_getStartTime(timer));

  for (iter = wbTimer_getHead(timer); iter != nullptr;
       iter = wbTimerNode_getNext(iter)) {
    if (!wbTimerNode_stoppedQ(iter)) {
      wbTimerNode_setEndTime(iter, currentTime);
      wbTimerNode_setElapsedTime(iter, currentTime -
                                           wbTimerNode_getStartTime(iter));
    }
    elems.push_back(wbTimerNode_toJSONObject(iter));
  }
  return json11::Json(elems);
}

string wbTimer_toJSON(wbTimer_t timer) {
  if (timer == nullptr) {
    return "";
  } else if (WB_USE_JSON11) {
    json11::Json json = wbTimer_toJSONObject(timer);
    return json.string_value();
  } else {
    stringstream ss;
    wbTimerNode_t iter;
    uint64_t currentTime;

    currentTime = getTime();

    wbTimer_setEndTime(timer, currentTime);
    wbTimer_setElapsedTime(timer,
                           currentTime - wbTimer_getStartTime(timer));

    for (iter = wbTimer_getHead(timer); iter != nullptr;
         iter = wbTimerNode_getNext(iter)) {
      if (!wbTimerNode_stoppedQ(iter)) {
        wbTimerNode_setEndTime(iter, currentTime);
        wbTimerNode_setElapsedTime(
            iter, currentTime - wbTimerNode_getStartTime(iter));
      }
      ss << wbTimerNode_toJSON(iter);
      if (wbTimerNode_getNext(iter) != nullptr) {
        ss << ",\n";
      }
    }

    return ss.str();
  }
}

string wbTimer_toJSON() {
  return wbTimer_toJSON(_timer);
}

string wbTimer_toXML(wbTimer_t timer) {
  if (timer == nullptr) {
    return "";
  } else {
    stringstream ss;
    wbTimerNode_t iter;
    uint64_t currentTime;

    currentTime = getTime();

    wbTimer_setEndTime(timer, currentTime);
    wbTimer_setElapsedTime(timer,
                           currentTime - wbTimer_getStartTime(timer));

    ss << "<timer>\n";
    ss << "<start_time>" << wbTimer_getStartTime(timer)
       << "</start_time>\n";
    ss << "<end_time>" << wbTimer_getEndTime(timer) << "</end_time>\n";
    ss << "<elapsed_time>" << wbTimer_getElapsedTime(timer)
       << "</elapsed_time>\n";
    ss << "<elements>\n";
    for (iter = wbTimer_getHead(timer); iter != nullptr;
         iter = wbTimerNode_getNext(iter)) {
      if (!wbTimerNode_stoppedQ(iter)) {
        wbTimerNode_setEndTime(iter, currentTime);
        wbTimerNode_setElapsedTime(
            iter, currentTime - wbTimerNode_getStartTime(iter));
      }
      ss << wbTimerNode_toXML(iter);
    }
    ss << "</elements>\n";
    ss << "</timer>\n";

    return ss.str();
  }
}

string wbTimer_toXML() {
  return wbTimer_toXML(_timer);
}

wbTimer_t wbTimer_new(void) {
  wbTimer_t timer = wbNew(struct st_wbTimer_t);
  wbTimer_setId(timer, uuid());
  wbTimer_setSessionId(timer, sessionId());
  wbTimer_setLength(timer, 0);
  wbTimer_setHead(timer, NULL);
  wbTimer_setTail(timer, NULL);
  wbTimer_setStartTime(timer, getTime());
  wbTimer_setEndTime(timer, 0);
  wbTimer_setElapsedTime(timer, 0);

  return timer;
}

static inline wbTimerNode_t _findParent(wbTimer_t timer) {
  wbTimerNode_t iter;

  for (iter = wbTimer_getTail(timer); iter != nullptr;
       iter = wbTimerNode_getPrevious(iter)) {
    if (!wbTimerNode_stoppedQ(iter)) {
      return iter;
    }
  }
  return NULL;
}

static inline void _insertIntoList(wbTimer_t timer, wbTimerNode_t node) {
  if (wbTimer_emptyQ(timer)) {
    wbTimer_setHead(timer, node);
    wbTimer_setTail(timer, node);
  } else {
    wbTimerNode_t end = wbTimer_getTail(timer);
    wbTimer_setTail(timer, node);
    wbTimerNode_setNext(end, node);
    wbTimerNode_setPrevious(node, end);
  }
  wbTimer_incrementLength(timer);
}

wbTimerNode_t wbTimer_start(wbTimerKind_t kind, const char *file,
                            const char *fun, int line) {
  int id;
  uint64_t currentTime;
  wbTimerNode_t node;
  wbTimerNode_t parent;

  // wb_init(NULL, NULL);

  currentTime = getTime();

  id = wbTimer_getLength(_timer);

  node = wbTimerNode_new(id, kind, file, fun, line);

  parent = _findParent(_timer);
  _insertIntoList(_timer, node);

  wbTimerNode_setStartTime(node, currentTime);
  wbTimerNode_setParent(node, parent);
  if (parent != nullptr) {
    wbTimerNode_setLevel(node, wbTimerNode_getLevel(parent) + 1);
  }

  return node;
}

wbTimerNode_t wbTimer_start(wbTimerKind_t kind, string msg,
                            const char *file, const char *fun, int line) {
  wbTimerNode_t node = wbTimer_start(kind, file, fun, line);
  wbTimerNode_setMessage(node, wbString_duplicate(msg));
  return node;
}

static inline wbTimerNode_t _findNode(wbTimer_t timer, wbTimerKind_t kind,
                                      string msg) {
  wbTimerNode_t iter;

  for (iter = wbTimer_getTail(timer); iter != nullptr;
       iter = wbTimerNode_getPrevious(iter)) {
    if (msg == "") {
      if (!wbTimerNode_stoppedQ(iter) &&
          wbTimerNode_getKind(iter) == kind) {
        return iter;
      }
    } else {
      if (!wbTimerNode_stoppedQ(iter) &&
          wbTimerNode_getKind(iter) == kind &&
          msg == wbTimerNode_getMessage(iter)) {
        return iter;
      }
    }
  }
  return NULL;
}

void wbTimer_stop(wbTimerKind_t kind, string msg, const char *file,
                  const char *fun, int line) {
  uint64_t currentTime;
  wbTimerNode_t node;

  currentTime = getTime();

  node = _findNode(_timer, kind, msg);

  wbAssert(node != nullptr);
  if (node == nullptr) {
    return;
  }

  wbTimerNode_setEndTime(node, currentTime);
  wbTimerNode_setElapsedTime(node,
                             currentTime - wbTimerNode_getStartTime(node));
  wbTimerNode_setEndLine(node, line);
  wbTimerNode_setEndFunction(node, fun);
  wbTimerNode_setEndFile(node, file);
  wbTimerNode_setStoppedQ(node, wbTrue);

#ifdef wbLogger_printOnLog
  if (wbLogger_printOnLog && node) {
    json11::Json json = json11::Json::object{
        {"type", "timer"},
        {"id", wbTimerNode_getId(node)},
        {"session_id", wbTimerNode_getSessionId(node)},
        {"data", wbTimerNode_toJSONObject(node)}};
    std::cout << json.dump() << std::endl;
  }
#endif /* wbLogger_printOnLog */
  return;
}

void wbTimer_stop(wbTimerKind_t kind, const char *file, const char *fun,
                  int line) {
  wbTimer_stop(kind, "", file, fun, line);
}
