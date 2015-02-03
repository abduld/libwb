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
file(GLOB  wbhdr *.hpp *.h)
file(GLOB  wbsrc *.cpp *.c)

# Simple macro to ease creation of MP executables.
macro (add_mp_executable name sources)
    cuda_add_executable(${name} ${sources} ${wbhdr} ${wbsrc})
    if(UNIX)
        target_link_libraries(${name} ${LINK_LIBRARIES})
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
