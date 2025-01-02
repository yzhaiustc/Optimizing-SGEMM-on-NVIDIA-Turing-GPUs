BINARY_NAME = sgemm_gpu
CUDA_PATH   = /usr/local/cuda
CC			= $(CUDA_PATH)/bin/nvcc
CFLAGS		= -O3 -std=c++11
LDFLAGS		= -L$(CUDA_PATH)/lib64 -lcudart -lcublas
INCFLAGS	= -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc


SRC			= $(wildcard *.cu)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)