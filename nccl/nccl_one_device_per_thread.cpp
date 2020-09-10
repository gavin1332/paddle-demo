#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>
#include <thread>
#include <sys/time.h>

const int g_comm_size = 32*1024*1024;

const int g_nranks = 8;

const int g_nloops = 10;

ncclUniqueId g_nccl_id;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int func(int rank, int sleep_time) {
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  //picking a GPU based on rank, allocate device buffers
  CUDACHECK(cudaSetDevice(rank));
  CUDACHECK(cudaMalloc(&sendbuff, g_comm_size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, g_comm_size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, g_nranks, g_nccl_id, rank));

  struct timeval start;
  struct timeval end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < g_nloops; ++i) {
    //communicating using NCCL
    if (rank == 0 && sleep_time > 0) sleep(sleep_time);
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, g_comm_size,
          ncclFloat, ncclSum, comm, s));
  }

  gettimeofday(&end, NULL);
  float time_use = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
  printf("[Rank %d] Success time used %.2fms\n", rank, time_use);

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  //finalizing NCCL
  ncclCommDestroy(comm);

  return 0;
}


int main(int argc, char* argv[]) {
  int sleep_time = 0;
  if (argc > 1) {
    sleep_time = atoi(argv[1]);
  }

  //get NCCL unique ID
  ncclGetUniqueId(&g_nccl_id);

  std::thread t[g_nranks];
  for (int i = 0; i < g_nranks; ++i) {
    t[i] = std::thread(func, i, sleep_time);
  }

  for (int i = 0; i < g_nranks; ++i) {
    t[i].join();
  }
}
