CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -I/usr/include/OpenMesh
LDFLAGS = -lOpenMeshCore -lOpenMeshTools

TARGET = main
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
