##########################################
# Options
##########################################
WB_LIB_PATH=$(CURDIR)/lib
WB_SRC_PATH=$(CURDIR)

##########################################
##########################################

CXX=g++
DEFINES=
CXX_FLAGS=-fPIC -Wno-unused-function -x c++ -O3 -g -std=c++11 -Wall -Wno-unused-function -pedantic -I . -I $(WB_SRC_PATH) $(DEFINES)
LIBS=-lm -lstdc++

##########################################
##########################################

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

##########################################
##########################################

SOURCES :=  $(WB_SRC_PATH)/wbArg.cpp               \
						$(WB_SRC_PATH)/wbCUDA.cpp              \
						$(WB_SRC_PATH)/wbDirectory.cpp         \
						$(WB_SRC_PATH)/wbDataGenerator.cpp     \
						$(WB_SRC_PATH)/wbExit.cpp              \
						$(WB_SRC_PATH)/wbExport.cpp            \
						$(WB_SRC_PATH)/wbFile.cpp              \
						$(WB_SRC_PATH)/wbImage.cpp             \
						$(WB_SRC_PATH)/wbImport.cpp            \
						$(WB_SRC_PATH)/wbInit.cpp              \
						$(WB_SRC_PATH)/wbLogger.cpp            \
						$(WB_SRC_PATH)/wbMPI.cpp               \
						$(WB_SRC_PATH)/wbPPM.cpp               \
						$(WB_SRC_PATH)/wbSolution.cpp          \
						$(WB_SRC_PATH)/wbSparse.cpp            \
						$(WB_SRC_PATH)/wbTimer.cpp

OBJECTS = $(SOURCES:.cpp=.o)

TESTS :=  $(WB_SRC_PATH)/wb_test.cpp               \
					$(WB_SRC_PATH)/wbDataGenerator_test.cpp

TESTOBJECTS = $(TESTS:.cpp=.o)

##############################################
# OUTPUT
##############################################

.PHONY: all
.SUFFIXES: .o .cpp
all: libwb.so

.cpp.o:
	$(CXX) $(DEFINES) $(CXX_FLAGS) -c -o $@ $<

libwb.so: $(OBJECTS)
	mkdir -p $(WB_LIB_PATH)
	$(CXX) -fPIC -shared $(LIBS) -o $(WB_LIB_PATH)/$@ $(OBJECTS)

libwb.a: $(OBJECTS)
	mkdir -p $(WB_LIB_PATH)
	ar rcs -o $(WB_LIB_PATH)/$@ $(OBJECTS)

test: $(TESTOBJECTS) $(OBJECTS)
	$(CXX) -fPIC $(LIBS) -o $@ $(TESTOBJECTS) $(OBJECTS)


clean:
	rm -fr $(ARCH)
	-rm -f $(EXES) *.o *~
