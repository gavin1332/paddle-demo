OUTPUT of `sh run_dy.sh`

```
-----------  Configuration Arguments -----------
cluster_node_ips: 127.0.0.1
log_dir: None
node_ip: 127.0.0.1
print_config: True
selected_gpus: 0,1
started_port: None
training_script: dy_collective.py
training_script_args: []
use_paddlecloud: False
------------------------------------------------
trainers_endpoints: 127.0.0.1:16978,127.0.0.1:22447 , node_id: 0 , current_node_ip: 127.0.0.1 , num_nodes: 1 , node_ips: ['127.0.0.1'] , nranks: 2
I0326 17:09:29.938866 141041 nccl_context.cc:127] init nccl context nranks: 2 local rank: 0 gpu id: 0
I0326 17:09:29.938895 141042 nccl_context.cc:127] init nccl context nranks: 2 local rank: 1 gpu id: 1
W0326 17:09:31.341827 141042 device_context.cc:237] Please NOTE: device: 1, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0326 17:09:31.355130 141041 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0326 17:09:31.380182 141042 device_context.cc:245] device: 1, cuDNN Version: 7.4.
W0326 17:09:31.380182 141041 device_context.cc:245] device: 0, cuDNN Version: 7.4.

local_rank: 0
nranks: 2
current endpoint: 127.0.0.1:16978
input: [1 1]
allreduce: [3 3]
c_allreduce: [3 3]
c_allgather: [1 1 2 2]
c_reducescatter: [2 2]
c_broadcast: [1 1]

local_rank: 1
nranks: 2
current endpoint: 127.0.0.1:22447
input: [1 1]
allreduce: [3 3]
c_allreduce: [3 3]
c_allgather: [1 1 2 2]
c_reducescatter: [4 4]
c_broadcast: [1 1]
```

OUTPUT of `sh run_pe.sh`

```
-----------  Configuration Arguments -----------
cluster_node_ips: 127.0.0.1
log_dir: None
node_ip: 127.0.0.1
print_config: True
selected_gpus: 0,1
started_port: None
training_script: pe_collective.py
training_script_args: []
use_paddlecloud: False
------------------------------------------------
trainers_endpoints: 127.0.0.1:54168,127.0.0.1:11396 , node_id: 0 , current_node_ip: 127.0.0.1 , num_nodes: 1 , node_ips: ['127.0.0.1'] , nranks: 2
server not ready, wait 3 sec to retry...
not ready endpoints:['127.0.0.1:11396']
W0326 17:11:14.573264 179012 device_context.cc:237] Please NOTE: device: 1, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0326 17:11:14.614727 179012 device_context.cc:245] device: 1, cuDNN Version: 7.4.
server not ready, wait 3 sec to retry...
not ready endpoints:['127.0.0.1:11396']
E0326 17:11:16.476913899  182169 tcp_server_posix.cc:64]     check for SO_REUSEPORT: {"created":"@1585213876.476898971","description":"SO_REUSEPORT unavailable on compiling system","file":"src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":163}
I0326 17:11:16.477905 182169 grpc_server.cc:477] Server listening on 127.0.0.1:11396 successful, selected port: 11396
W0326 17:11:20.385125 179001 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0326 17:11:20.390413 179001 device_context.cc:245] device: 0, cuDNN Version: 7.4.
I0326 17:11:21.960572 179001 rpc_client.h:107] init rpc client with trainer_id 0
I0326 17:11:21.966527 179012 rpc_server.cc:28] RPCServer ShutDown 
W0326 17:11:21.967173 182180 grpc_server.cc:604] CompletionQueue RequestSend shutdown!
W0326 17:11:21.967173 182182 grpc_server.cc:604] CompletionQueue RequestSend shutdown!
W0326 17:11:21.967186 182183 grpc_server.cc:604] CompletionQueue RequestSend shutdown!
W0326 17:11:21.967182 182181 grpc_server.cc:604] CompletionQueue RequestSend shutdown!
W0326 17:11:21.967176 182184 grpc_server.cc:604] CompletionQueue RequestSend shutdown!
I0326 17:11:21.969600 179001 parallel_executor.cc:481] The Program will be executed on CUDA using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0326 17:11:21.969738 179012 parallel_executor.cc:481] The Program will be executed on CUDA using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
W0326 17:11:22.320742 179012 fuse_all_reduce_op_pass.cc:74] Find all_reduce operators: 2. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 1.
W0326 17:11:22.320775 179001 fuse_all_reduce_op_pass.cc:74] Find all_reduce operators: 2. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 1.
I0326 17:11:22.320899 179012 build_strategy.cc:376] SeqOnlyAllReduceOps:1, num_trainers:2
I0326 17:11:22.320931 179001 build_strategy.cc:376] SeqOnlyAllReduceOps:1, num_trainers:2
I0326 17:11:22.322233 179012 parallel_executor.cc:333] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0326 17:11:22.322233 179001 parallel_executor.cc:333] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0326 17:11:22.322513 179012 parallel_executor.cc:401] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
I0326 17:11:22.322528 179001 parallel_executor.cc:401] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
rank: 0
allgather: [[1.]
 [1.]]
 allreduce: [[2.]]

 rank: 1
 allgather: [[1.]
  [1.]]
  allreduce: [[2.]]
```
