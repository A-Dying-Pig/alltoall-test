.PNONY: compile run clean debug

ROCM_PATH = /opt/rocm-6.2.0
TARGET = all2all_tester
HIPCC = $(ROCM_PATH)/bin/hipcc
MPI_HOME = /usr/local/mpi
CFLAGS = -O3 -std=c++11
RCCL_HOME ?= /root/rccl-alltoall/rccl/build

INC =  -I$(RCCL_HOME)/ -I$(RCCL_HOME)/include -I$(RCCL_HOME)/hipify/src/include
INC += -I$(ROCM_PATH)/include
INC += -I$(ROCM_PATH)/include/hip
INC += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi

BUILD_DIR = $(shell pwd)/build


LIB = -lpthread  -L$(RCCL_HOME) -L$(RCCL_HOME)/lib
# LIB += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
LIB += -L${MPI_HOME}/lib -lmpi
LIB += -Wl,-rpath,$(RCCL_HOME) -L$(RCCL_HOME) -lrccl

SOURCES = test_alltoall.cpp
SRC_DIR = $(RCCL_HOME)/hipify/src/fast_alltoall
INC_DIR = $(RCCL_HOME)/hipify/src/include/fast_alltoall

OBJ = $(BUILD_DIR)/alltoall_matrix.o
OBJ += $(BUILD_DIR)/alltoall_algorithm.o
OBJ += $(BUILD_DIR)/alltoall_local_scheduler.o
OBJ += $(BUILD_DIR)/alltoall_global_scheduler.o

compile: test_alltoall.cpp $(INC_DIR)/alltoall_define.h $(BUILD_DIR)/alltoall_matrix.o $(BUILD_DIR)/alltoall_algorithm.o $(BUILD_DIR)/alltoall_local_scheduler.o $(BUILD_DIR)/alltoall_global_scheduler.o
	@echo "Building AllToAll Tester"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(HIPCC) $(CFLAGS) $(INC) $(OBJ) $(SOURCES) -o $(BUILD_DIR)/$(TARGET) $(LIB)
	@echo "Building successful"

$(BUILD_DIR)/alltoall_matrix.o: ${SRC_DIR}/alltoall_matrix.cc $(INC_DIR)/alltoall_matrix.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_matrix.cc -o $(BUILD_DIR)/alltoall_matrix.o

$(BUILD_DIR)/alltoall_algorithm.o: ${SRC_DIR}/alltoall_algorithm.cc $(INC_DIR)/alltoall_algorithm.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_algorithm.cc -o $(BUILD_DIR)/alltoall_algorithm.o

$(BUILD_DIR)/alltoall_local_scheduler.o: ${SRC_DIR}/alltoall_local_scheduler.cc $(INC_DIR)/alltoall_local_scheduler.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_local_scheduler.cc -o $(BUILD_DIR)/alltoall_local_scheduler.o

$(BUILD_DIR)/alltoall_global_scheduler.o: ${SRC_DIR}/alltoall_global_scheduler.cc $(INC_DIR)/alltoall_global_scheduler.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_global_scheduler.cc -o $(BUILD_DIR)/alltoall_global_scheduler.o


run:
	mpirun --allow-run-as-root -np 8 $(BUILD_DIR)/$(TARGET)



clean:
	@echo "Cleaning FastAll2All"
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

debug:
	sudo gdb --args $(BUILD_DIR)/$(TARGET)

