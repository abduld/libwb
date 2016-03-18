cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(wb)

set(current_dir "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/cotire.cmake")
  file(DOWNLOAD "https://raw.githubusercontent.com/sakra/cotire/master/CMake/cotire.cmake" "${CMAKE_CURRENT_BINARY_DIR}/cotire.cmake")
endif()



include(${current_dir}/sources.cmake)

set(WBLIB_STATIC ${WBLIB})
set(WBLIB_SHARED lib${WBLIB})

add_library(${WBLIB_STATIC} STATIC ${LIBWB_SOURCE_FILES} )
add_library(${WBLIB_SHARED} SHARED ${LIBWB_SOURCE_FILES} )
set_property(TARGET ${WBLIB_STATIC} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${WBLIB_SHARED} PROPERTY CXX_STANDARD 11)
if (UNIX)
  set_target_properties(${WBLIB_SHARED} PROPERTIES OUTPUT_NAME ${WBLIB_STATIC})
endif (UNIX)
set_property(TARGET ${WBLIB} PROPERTY CXX_STANDARD 11)
