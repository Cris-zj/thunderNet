CC:=nvcc
FLAG:= -g -G -std=c++11 -shared --compiler-options '-fPIC' -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_EXTENSION_NAME=libpsroi
LIBARAY:= cudart
LIB_PATH := /usr/local/cuda/lib64
INCLUDE := /home/huangeryu/.local/lib/python3.6/site-packages/torch/include \
			/home/huangeryu/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include \
			/home/huangeryu/.local/lib/python3.6/site-packages/torch/include/TH \
			/home/huangeryu/.local/lib/python3.6/site-packages/torch/include/THC \
			/usr/local/cuda/include \
			/usr/include/python3.6m 
TARGET := libpsroi.so
SOURCE := psroi.cpp psroi.cu

PSROI:
	$(CC) $(FLAG) $(addprefix -isystem ,$(INCLUDE))  $(addprefix -L,$(LIB_PATH)) $(addprefix -l,$(LIBARAY))  -o $(TARGET) $(SOURCE)
