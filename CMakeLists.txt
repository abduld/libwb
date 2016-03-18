cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(libwb)

set(current_dir "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT EXISTS "${current_dir}/cotire.cmake")
  file(DOWNLOAD "https://raw.githubusercontent.com/sakra/cotire/master/CMake/cotire.cmake" "${current_dir}/cotire.cmake")
endif()



include(${current_dir}/sources.cmake)

add_library(${WBLIB} ${LIBWB_SOURCE_FILES} STATIC)
add_library(${WBLIB} ${LIBWB_SOURCE_FILES} SHARED)
set_property(TARGET ${WBLIB} PROPERTY CXX_STANDARD 11)
