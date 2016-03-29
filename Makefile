##########################################
# Options
##########################################
WB_LIB_PATH=$(CURDIR)/lib
WB_SRC_PATH=$(CURDIR)

##########################################
##########################################

DEFINES=
CXX_FLAGS=-fPIC -Wno-unused-function -Wno-dollar-in-identifier-extension \
					-x c++ -O3 -g -std=c++11 -Wall -Wno-unused-function -pedantic \
					-I . -I $(WB_SRC_PATH) $(DEFINES) 
LIBS=-lm -lstdc++ -L $(WB_LIB_PATH)

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
	$(CXX) -fPIC -shared -o $(WB_LIB_PATH)/$@ $(OBJECTS) $(LIBS)

libwb.a: $(OBJECTS)
	mkdir -p $(WB_LIB_PATH)
	ar rcs -o $(WB_LIB_PATH)/$@ $(OBJECTS)

test: libwb.so
	$(CXX) $(DEFINES) $(CXX_FLAGS) -fPIC -o $@ $(TESTS) -lwb $(LIBS)


clean:
	rm -fr $(ARCH) test
	-rm -f $(EXES) *.o *~