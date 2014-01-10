
libwb
=====



# Compiling and Running on Linux and Windows
# by Tran Minh Quan

This is a tutorial explains how to compile and run your Machine
Problems (MPs) offline **without separating on building libwb.**

_Caution: **If you don't have NVIDIA GPUs ([CUDA Capable GPU](https://developer.nvidia.com/cuda-gpus)s) on your local machine, you cannot run the executable binaries.**_

First, regardless your platform is, please install CUDA 5.5
and Cmake 2.8 ([](http://www.cmake.org/)[](http://www.cmake.org/)[](http://www.cmake.org/)[](http://www.cmake.org/)[http://www.cmake.org/](http://www.cmake.org/)) , then set the path appropriate to these things (linux).

Check out the source codes (only skeleton codes) for MPs as
following

[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[https://github.com/hvcl/hetero13](https://github.com/hvcl/hetero13)

    git clone https://github.com/abduld/libwb

1\. If you are under Linux environment, you should use gcc lower than 4.7 (mine is 4.4.7). 
Ortherwise, it will not be compatible with nvcc

    cd libwb
    ls
    mkdir build
    cd build/
    cmake ..
    make -j4
    ./MP0

2\. If you are under Windows environment

Open Cmake Gui:

    Where is the source code: {libwb}/
    Where to build the binary: {libwb}/build

![image](https://coursera-forum-screenshots.s3.amazonaws.com/5d/d77a10785611e3ae687ff4063e578b/1.png) 

Press Configure, Yes and choose your compiler (in this case Visual
Studio 10 (32 bit) or Visual Studio 10 Win64 (64 bit), then press Finish

![image](https://coursera-forum-screenshots.s3.amazonaws.com/75/ee29f0785611e3ae687ff4063e578b/2.png) 

![image](https://coursera-forum-screenshots.s3.amazonaws.com/e5/1e0fc0785611e3ae687ff4063e578b/3.png) 

Press Configure one more time and generate

![image](https://coursera-forum-screenshots.s3.amazonaws.com/11/315360785711e3ae687ff4063e578b/4.png) 

Open your generated folder and Double click on libwb.sln

![image](https://coursera-forum-screenshots.s3.amazonaws.com/3a/5da3b0785711e3ae687ff4063e578b/5.png) 

Right click to MP0 and click "set as startup project"

Press Ctrl F5

Whenever you do the MPs, change the MP accordingly.

Best regards,

P/s1: Sorry about the name of project, it should be hetero14 but I forget that we are already in the new year. My appologize :-P 

P/s2: If you are using MAC, please consider reading this link and modify your CMakeLists.txt
[https://class.coursera.org/hetero-002/forum/thread?thread\_id=83](https://class.coursera.org/hetero-002/forum/thread?thread_id=83)

