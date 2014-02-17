############################################################################################
# <summary> CMakeLists.txt for libWB and CUDA.                </summary>
# <date>    2013-01-07                                </date>
# <author>  Tran Minh Quan. edit by Johannes Kast, Michael Sarahan      </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com </email>
############################################################################################
project(libwb)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
find_package(CUDA REQUIRED)
# collect source files
 
if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect 
    # whether to build x64.  We tell it to support most devices, though, 
    # to make sure more people can easily run class code without knowing 
    # about this compiler argument
    set(CUDA_NVCC_FLAGS "")
 
    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    endif (UNIX)
 
    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
  include_directories(${CMAKE_SOURCE_DIR})
  include_directories(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
 
if (UNIX)
    include(CheckLibraryExists)
    check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME )
    if(NOT HAVE_CLOCK_GETTIME)
        message(FATAL_ERROR "clock_gettime not found")
    endif(NOT HAVE_CLOCK_GETTIME)
endif(UNIX)
 
file( GLOB  wbhdr *.hpp *.h )
file( GLOB  wbsrc *.cpp *.c )
 
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
 
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
