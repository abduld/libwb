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

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
# set(COMPUTE_CAPABILITY 35)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--gpu-architecture;sm_${COMPUTE_CAPABILITY})

# Set warning levels for host compilation
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wextra")
endif ()

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif()
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else ()
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -isystem ${CUDA_INCLUDE_DIRS}")
    endif ()
endif ()

# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

# Set the correct linker options on UNIX platforms
if (UNIX)
    set(LINK_LIBRARIES "-lm")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else ()
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else ()
            message(FATAL_ERROR "clock_gettime() not found")
        endif ()
    endif ()
endif ()

# Collect source files
file(GLOB WBHDR *.hpp *.h)
file(GLOB WBSRC *.cpp *.c)

set(EXECUTABLES mp0 mp1 mp2 mp3 mp4 mp5 mp6 mp11 mp12)
foreach (EXECUTABLE ${EXECUTABLES})
    string(TOUPPER "${EXECUTABLE}" LABEL)
    CUDA_ADD_EXECUTABLE(${LABEL} skel/${EXECUTABLE}.cu ${WBHDR} ${WBSRC})
    if (UNIX)
        target_link_libraries(${LABEL} ${LINK_LIBRARIES})
    endif ()
endforeach ()
