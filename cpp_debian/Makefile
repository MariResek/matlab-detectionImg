CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++14 -fPIC -fopenmp -Wno-unused-variable -Wno-unused-but-set-variable

GEN_DIR  := yolov2_detect
INCLUDES := -I. -I$(GEN_DIR)

GEN_SRCS := $(wildcard $(GEN_DIR)/*.cpp)
SRCS     := main.cpp $(GEN_SRCS)
OBJS     := $(SRCS:.cpp=.o)

BIN := detect

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread -fopenmp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(BIN)

.PHONY: all clean
