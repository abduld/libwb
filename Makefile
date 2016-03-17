##########################################
# Options
##########################################
WB_LIB_PATH=$(CURDIR)/lib
WB_SRC_PATH=$(CURDIR)

##########################################
##########################################

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

SOURCES := $(shell find $(WB_SRC_PATH) ! -name "*_test.cpp" -name "*.cpp")
TESTS :=  $(shell find $(WB_SRC_PATH) -name "*_test.cpp")

OBJECTS = $(SOURCES:.cpp=.o)
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
