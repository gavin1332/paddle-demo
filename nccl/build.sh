NCCL_INCLUDE=/home/users/liuyi05/opt/nccl237c92/include
NCCL_LIB=/home/users/liuyi05/opt/nccl237c92/lib
g++ -std=c++11 -o nccl_one_device_per_thread nccl_one_device_per_thread.cpp -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -I$NCCL_INCLUDE -L$NCCL_LIB -lcudart -lnccl 
