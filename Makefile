##########################################
# Options
##########################################
WB_LIB_PATH=$(CURDIR)/lib
WB_SRC_PATH=$(CURDIR)
CXX=g++

##########################################
##########################################

DEFINES=-DWB_USE_COURSERA -DWB_USE_JSON11=0
CXX_FLAGS=-fpic -O3 -g -std=c++11 -I . -I $(WB_SRC_PATH) # -I /usr/local/cuda/include -L /usr/local/cuda/lib64 
LIBS=-lm -std=c++11 -L $(WB_LIB_PATH) # -lcuda

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
	$(CXX) -shared -o $(WB_LIB_PATH)/$@ $(OBJECTS) $(LIBS)

libwb.a: $(OBJECTS)
	mkdir -p $(WB_LIB_PATH)
	ar rcs -o $(WB_LIB_PATH)/$@ $(OBJECTS)

test: libwb.so
	$(CXX) $(DEFINES) $(CXX_FLAGS) -o $@ $(TESTS) -lwb $(LIBS)


clean:
	rm -fr lib test
	-rm -f $(EXES) *.o *~
