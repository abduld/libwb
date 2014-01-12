############################################################################################
# <summary> CMakeLists.txt for libWB and CUDA.                </summary>
# <date>    2013-01-07                                </date>
# <author>  Tran Minh Quan. edit by Johannes Kast, Michael Sarahan      </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com </email>
############################################################################################
project(hetero13)
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

file( GLOB  wbhdr *.hpp *.h )
file( GLOB  wbsrc *.cpp *.c )

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(MP0 skel/mp0.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP1 skel/mp1.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2 skel/mp2.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3 skel/mp3.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP4 skel/mp4.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5 skel/mp5.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP6 skel/mp6.cu ${wbhdr} ${wbsrc})