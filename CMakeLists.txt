############################################################################################
# <summary> CMakeLists.txt for libWB and CUDA.                </summary>
# <date>    2013-01-07                                </date>
# <author>  Tran Minh Quan. edit by Johannes Kast, Michael Sarahan      </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com </email>
############################################################################################
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libwb)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

# Set warning levels for host compilation
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()

# add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

if (UNIX)
    include(CheckLibraryExists)
    check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME )
    if(NOT HAVE_CLOCK_GETTIME)
        message(FATAL_ERROR "clock_gettime not found")
    endif(NOT HAVE_CLOCK_GETTIME)
endif(UNIX)

# Collect source files
file(GLOB  wbhdr *.hpp *.h)
file(GLOB  wbsrc *.cpp *.c)

# Simple macro to ease creation of MP executables.
macro (add_mp_executable name sources)
    cuda_add_executable(${name} ${sources} ${wbhdr} ${wbsrc})
    if(UNIX)
        target_link_libraries(${name} ${LIBRT_LIBRARIES})
    endif(UNIX)
endmacro()

add_mp_executable(MP0 skel/mp0.cu)
add_mp_executable(MP1 skel/mp1.cu)
add_mp_executable(MP2 skel/mp2.cu)
add_mp_executable(MP3 skel/mp3.cu)
add_mp_executable(MP4 skel/mp4.cu)
add_mp_executable(MP5 skel/mp5.cu)
add_mp_executable(MP6 skel/mp6.cu)
add_mp_executable(MP11 skel/mp11.cu)
add_mp_executable(MP12 skel/mp12.cu)
