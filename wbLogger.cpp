
#include "wb.h"

wbLogger_t _logger = NULL;

static inline wbBool wbLogEntry_hasNext(wbLogEntry_t elem) {
  return wbLogEntry_getNext(elem) != NULL;
}

static inline wbLogEntry_t wbLogEntry_new() {
  wbLogEntry_t elem;

  elem = wbNew(struct st_wbLogEntry_t);

  wbLogEntry_setId(elem, uuid());
  wbLogEntry_setSessionId(elem, sessionId());
  wbLogEntry_setMessage(elem, NULL);
  wbLogEntry_setMPIRank(elem, wbMPI_getRank());
  wbLogEntry_setTime(elem, _hrtime());

  wbLogEntry_setLevel(elem, wbLogLevel_TRACE);

  wbLogEntry_setNext(elem, NULL);

  wbLogEntry_setLine(elem, -1);
  wbLogEntry_setFile(elem, NULL);
  wbLogEntry_setFunction(elem, NULL);

  return elem;
}

static inline wbLogEntry_t
wbLogEntry_initialize(wbLogLevel_t level, string msg, const char *file,
                      const char *fun, int line) {
  wbLogEntry_t elem;

  elem = wbLogEntry_new();

  wbLogEntry_setLevel(elem, level);

  wbLogEntry_setMessage(elem, wbString_duplicate(msg));

  wbLogEntry_setLine(elem, line);
  wbLogEntry_setFile(elem, file);
  wbLogEntry_setFunction(elem, fun);

  return elem;
}

static inline void wbLogEntry_delete(wbLogEntry_t elem) {
  if (elem != NULL) {
    if (wbLogEntry_getMessage(elem) != NULL) {
      wbFree(wbLogEntry_getMessage(elem));
    }
    wbDelete(elem);
  }
  return;
}

static inline const char *getLevelName(wbLogLevel_t level) {
  switch (level) {
    case wbLogLevel_unknown:
      return "Unknown";
    case wbLogLevel_OFF:
      return "Off";
    case wbLogLevel_FATAL:
      return "Fatal";
    case wbLogLevel_ERROR:
      return "Error";
    case wbLogLevel_WARN:
      return "Warn";
    case wbLogLevel_INFO:
      return "Info";
    case wbLogLevel_DEBUG:
      return "Debug";
    case wbLogLevel_TRACE:
      return "Trace";
  }
  return NULL;
}

static inline json11::Json wbLogEntry_toJSONObject(wbLogEntry_t elem) {
  json11::Json json = json11::Json::object{
      {"id", wbLogEntry_getId(elem)},
      {"session_id", wbLogEntry_getSessionId(elem)},
      {"mpi_rank", wbLogEntry_getMPIRank(elem)},
      {"level", getLevelName(wbLogEntry_getLevel(elem))},
      {"file", wbLogEntry_getFile(elem)},
      {"function", wbLogEntry_getFunction(elem)},
      {"line", wbLogEntry_getLine(elem)},
      {"time", wbLogEntry_getTime(elem)},
      {"message", wbLogEntry_getMessage(elem)},
  };
  return json;
}

static inline string wbLogEntry_toJSON(wbLogEntry_t elem) {
  if (elem == NULL) {
    return "";
  } else if (WB_USE_JSON11) {
    json11::Json json = wbLogEntry_toJSONObject(elem);
    return json.string_value();
  } else {
    stringstream ss;

    ss << "{\n";
    ss << wbString_quote("id") << ":"
       << wbString_quote(wbLogEntry_getId(elem)) << ",\n";
    ss << wbString_quote("session_id") << ":"
       << wbString_quote(wbLogEntry_getSessionId(elem)) << ",\n";
    ss << wbString_quote("mpi_rank") << ":"
       << wbString(wbLogEntry_getMPIRank(elem)) << ",\n";
    ss << wbString_quote("level") << ":"
       << wbString_quote(getLevelName(wbLogEntry_getLevel(elem))) << ",\n";
    ss << wbString_quote("message") << ":"
       << wbString_quote(wbLogEntry_getMessage(elem)) << ",\n";
    ss << wbString_quote("file") << ":"
       << wbString_quote(wbLogEntry_getFile(elem)) << ",\n";
    ss << wbString_quote("function") << ":"
       << wbString_quote(wbLogEntry_getFunction(elem)) << ",\n";
    ss << wbString_quote("line") << ":" << wbLogEntry_getLine(elem)
       << ",\n";
    ss << wbString_quote("time") << ":" << wbLogEntry_getTime(elem)
       << "\n";
    ss << "}";

    return ss.str();
  }
  return "";
}

static inline string wbLogEntry_toXML(wbLogEntry_t elem) {
  if (elem != NULL) {
    stringstream ss;

    ss << "<entry>\n";
    ss << "<type>"
       << "LoggerElement"
       << "</type>\n";
    ss << "<id>" << wbLogEntry_getId(elem) << "</id>\n";
    ss << "<session_id>" << wbLogEntry_getSessionId(elem)
       << "</session_id>\n";
    ss << "<level>" << wbLogEntry_getLevel(elem) << "</level>\n";
    ss << "<message>" << wbLogEntry_getMessage(elem) << "</message>\n";
    ss << "<file>" << wbLogEntry_getFile(elem) << "</file>\n";
    ss << "<function>" << wbLogEntry_getFunction(elem) << "</function>\n";
    ss << "<line>" << wbLogEntry_getLine(elem) << "</line>\n";
    ss << "<time>" << wbLogEntry_getTime(elem) << "</time>\n";
    ss << "</entry>\n";

    return ss.str();
  }
  return "";
}

wbLogger_t wbLogger_new() {
  wbLogger_t logger;

  logger = wbNew(struct st_wbLogger_t);

  wbLogger_setId(logger, uuid());
  wbLogger_setSessionId(logger, sessionId());
  wbLogger_setLength(logger, 0);
  wbLogger_setHead(logger, NULL);

  wbLogger_getLevel(logger) = wbLogLevel_TRACE;

  return logger;
}

static inline void _wbLogger_setLevel(wbLogger_t logger,
                                      wbLogLevel_t level) {
  wbLogger_getLevel(logger) = level;
}

static inline void _wbLogger_setLevel(wbLogLevel_t level) {
  _wbLogger_setLevel(_logger, level);
}

#define wbLogger_setLevel(level) _wbLogger_setLevel(wbLogLevel_##level)

void wbLogger_clear(wbLogger_t logger) {
  if (logger != NULL) {
    wbLogEntry_t tmp;
    wbLogEntry_t iter;

    iter = wbLogger_getHead(logger);
    while (iter != NULL) {
      tmp = wbLogEntry_getNext(iter);
      wbLogEntry_delete(iter);
      iter = tmp;
    }

    wbLogger_setLength(logger, 0);
    wbLogger_setHead(logger, NULL);
  }
}

void wbLogger_delete(wbLogger_t logger) {
  if (logger != NULL) {
    wbLogger_clear(logger);
    wbDelete(logger);
  }
  return;
}

void wbLogger_append(wbLogLevel_t level, string msg, const char *file,
                     const char *fun, int line) {
  wbLogEntry_t elem;
  wbLogger_t logger;

  wb_init(NULL, NULL);

  logger = _logger;

  if (wbLogger_getLevel(logger) < level) {
    return;
  }

  elem = wbLogEntry_initialize(level, msg, file, fun, line);

#ifdef wbLogger_printOnLog
  if (wbLogger_printOnLog) {
    if (level <= wbLogger_getLevel(logger) && elem) {
      json11::Json json = json11::Json::object{
          {"type", "logger"},
          {"id", wbLogEntry_getId(elem)},
          {"session_id", wbLogEntry_getSessionId(elem)},
          {"data", wbLogEntry_toJSONObject(elem)}};
      std::cout << json.dump() << std::endl;
    }
  }
#endif /* wbLogger_printOnLog */

  if (wbLogger_getHead(logger) == NULL) {
    wbLogger_setHead(logger, elem);
  } else {
    wbLogEntry_t prev = wbLogger_getHead(logger);

    while (wbLogEntry_hasNext(prev)) {
      prev = wbLogEntry_getNext(prev);
    }
    wbLogEntry_setNext(prev, elem);
  }

#if 0
  if (level <= wbLogger_getLevel(logger) && elem) {
    const char *levelName = getLevelName(level);

    fprintf(stderr, "= LOG: %s: %s (In %s:%s on line %d). =\n", levelName,
            wbLogEntry_getMessage(elem), wbLogEntry_getFile(elem),
            wbLogEntry_getFunction(elem), wbLogEntry_getLine(elem));
  }
#endif

  wbLogger_incrementLength(logger);

  return;
}

string wbLogger_toJSON() {
  return wbLogger_toJSON(_logger);
}

static json11::Json wbLogger_toJSONObject(wbLogger_t logger) {
  std::vector<json11::Json> elems{};

  if (logger != NULL) {
    wbLogEntry_t iter;
    stringstream ss;

    for (iter = wbLogger_getHead(logger); iter != NULL;
         iter = wbLogEntry_getNext(iter)) {
      elems.push_back(wbLogEntry_toJSONObject(iter));
    }
  }
  return json11::Json(elems);
}

string wbLogger_toJSON(wbLogger_t logger) {
  if (logger != NULL) {
    wbLogEntry_t iter;
    stringstream ss;

    for (iter = wbLogger_getHead(logger); iter != NULL;
         iter = wbLogEntry_getNext(iter)) {
      ss << wbLogEntry_toJSON(iter);
      if (wbLogEntry_getNext(iter) != NULL) {
        ss << ",\n";
      }
    }

    return ss.str();
  }
  return "";
}

string wbLogger_toXML() {
  return wbLogger_toXML(_logger);
}

string wbLogger_toXML(wbLogger_t logger) {
  if (logger != NULL) {
    wbLogEntry_t iter;
    stringstream ss;

    ss << "<logger>\n";
    ss << "<type>"
       << "Logger"
       << "</type>\n";
    ss << "<id>" << wbLogger_getId(logger) << "</id>\n";
    ss << "<session_id>" << wbLogger_getSessionId(logger)
       << "</session_id>\n";
    ss << "<elements>\n";
    for (iter = wbLogger_getHead(logger); iter != NULL;
         iter = wbLogEntry_getNext(iter)) {
      ss << wbLogEntry_toXML(iter);
    }
    ss << "</elements>\n";
    ss << "</logger>\n";

    return ss.str();
  }
  return "";
}
