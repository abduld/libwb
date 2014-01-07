

##########################################
# INPUT
##########################################
CXX=g++
DEFINES=-DWB_USE_CUDA
CUDA_INCLUDE=/usr/local/cuda-5.5/include
CXX_FLAGS=-fPIC -x c++ -O0 -g -I . -I $(CUDA_INCLUDE) -L $(HOME)/usr/lib -Wall  -I$(HOME)/usr/include $(DEFINES)
LIBS=-lm -lstdc++ -lrt -lcuda -L$(HOME)/usr/lib
ARCH=$(shell uname -s)-$(shell uname -i)

##########################################
##########################################

SOURCES :=  wbArg.cpp              \
			wbExit.cpp             \
			wbExport.cpp           \
			wbFile.cpp             \
			wbImage.cpp            \
			wbImport.cpp           \
			wbInit.cpp             \
			wbLogger.cpp           \
			wbPPM.cpp              \
			wbCUDA.cpp			   \
			wbSolution.cpp         \
			wbTimer.cpp


##############################################
# OUTPUT
##############################################

EXES = libwb.a libwb.so

.SUFFIXES : .o .cpp


OBJECTS = $(SOURCES:.cpp=.o)

##############################################
# OUTPUT
##############################################


.cpp.o:
	$(CXX) $(DEFINES) $(CXX_FLAGS) -c -o $@ $<


libwb.so:     $(OBJECTS)
	mkdir -p $(ARCH)
	$(CXX) -fPIC -shared $(LIBS) -o $(ARCH)/$@ $(OBJECTS)

libwb.a:     $(OBJECTS)
	mkdir -p $(ARCH)
	ar rcs -o $(ARCH)/$@ $(OBJECTS)

clean:
	rm -fr $(ARCH)
	-rm -f $(EXES) *.o *~


