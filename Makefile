.PNONY: compile run clean debug

ROCM_PATH = /opt/rocm-6.2.0
TARGET = all2all_tester
HIPCC = $(ROCM_PATH)/bin/hipcc
MPI_HOME = /usr/local/mpi
CFLAGS = -O3 -std=c++11
RCCL_HOME ?= /root/rccl-alltoall/rccl/build

INC = -I$(RCCL_HOME)/hipify/src/include/fast_alltoall
INC += -I$(RCCL_HOME)/include -I$(RCCL_HOME)/
INC += -I$(ROCM_PATH)/include
INC += -I$(ROCM_PATH)/include/hip
INC += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi

BUILD_DIR = $(shell pwd)/build


LIB = -L$(RCCL_HOME) -L$(RCCL_HOME)/librccl.so.1.0
LIB += -lpthread  -L$(RCCL_HOME)  -L$(NCCL_HOME)/lib
LIB += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
LIB += -L${MPI_HOME}/lib -lmpi -lrccl


SOURCES = test_alltoall.cpp

compile: test_alltoall.cpp
	@echo "Building AllToAll Tester"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(HIPCC) -g $(CFLAGS) $(INC) $(SOURCES) -o $(BUILD_DIR)/$(TARGET) $(LIB)
	@echo "Building successful"

run:
	$(BUILD_DIR)/$(TARGET)

clean:
	@echo "Cleaning FastAll2All"
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

debug:
	sudo gdb --args $(BUILD_DIR)/$(TARGET)

