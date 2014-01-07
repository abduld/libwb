

##########################################
# INPUT
##########################################
CXX=g++
DEFINES=-DWB_USE_CUDA
CUDA_INCLUDE=/usr/local/cuda-5.5/include
CXX_FLAGS=-fPIC -x c++ -O0 -g -I . -I $(CUDA_INCLUDE) -L $(HOME)/usr/lib -Wall  -I$(HOME)/usr/include $(DEFINES)
LIBS=-lm -lstdc++ -lrt -lcuda -L$(HOME)/usr/lib

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
	mkdir -p Linux-x86-64
	$(CXX) -fPIC -shared $(LIBS) -o Linux-x86-64/$@ $(OBJECTS)

libwb.a:     $(OBJECTS)
	mkdir Linux-x86-64
	ar rcs -o Linux-x86-64/$@ $(OBJECTS)

clean:
	rm -fr Linux-x86-64
	-rm -f $(EXES) *.o *~


