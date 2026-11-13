MPICXX    ?= mpicxx
EIGEN_INC ?= $(HOME)/local/include/eigen3

CXXFLAGS  ?= -O3 -DNDEBUG -std=c++17 -march=core-avx2 \
             -Wall -Wextra

LDLIBS    ?= -lstdc++fs

TARGET = code_vmc_1d
SRC    = code_vmc_1d.cpp

.PHONY: all clean check info

all: check $(TARGET)

check:
	@command -v $(MPICXX) >/dev/null 2>&1 || \
		{ echo "Error: $(MPICXX) was not found."; exit 1; }
	@test -f "$(EIGEN_INC)/Eigen/Dense" || \
		{ echo "Error: $(EIGEN_INC)/Eigen/Dense was not found."; exit 1; }
	@echo "MPI compiler : $$(command -v $(MPICXX))"
	@echo "Eigen include: $(EIGEN_INC)"

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -I$(EIGEN_INC) $< -o $@ $(LDLIBS)

info:
	@echo "MPICXX    = $(MPICXX)"
	@echo "EIGEN_INC = $(EIGEN_INC)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "LDLIBS    = $(LDLIBS)"
	@$(MPICXX) --version

clean:
	rm -f $(TARGET)
