MPICXX ?= /opt/local/bin/mpicxx
EIGEN_INC ?= /opt/local/include/eigen3
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -march=native -Wall -Wextra
TARGET = code_vmc_2d
SRC = code_vmc_2d.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -I$(EIGEN_INC) $< -o $@

clean:
	rm -f $(TARGET)
