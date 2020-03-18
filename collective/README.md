OUTPUT of `sh run.sh`

```
-----------  Configuration Arguments -----------
cluster_node_ips: 127.0.0.1
log_dir: None
node_ip: 127.0.0.1
print_config: True
selected_gpus: 0,1
started_port: None
training_script: collective.py
training_script_args: []
use_paddlecloud: False
------------------------------------------------
trainers_endpoints: 127.0.0.1:29282,127.0.0.1:57258 , node_id: 0 , current_node_ip: 127.0.0.1 , num_nodes: 1 , node_ips: ['127.0.0.1'] , nranks: 2
I0318 17:04:25.652648 31603 nccl_context.cc:127] init nccl context nranks: 2 local rank: 0 gpu id: 0
I0318 17:04:25.652660 31608 nccl_context.cc:127] init nccl context nranks: 2 local rank: 1 gpu id: 1
W0318 17:04:27.295740 31603 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0318 17:04:27.307015 31608 device_context.cc:237] Please NOTE: device: 1, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0318 17:04:27.333683 31603 device_context.cc:245] device: 0, cuDNN Version: 7.4.
W0318 17:04:27.333700 31608 device_context.cc:245] device: 1, cuDNN Version: 7.4.

local_rank: 0
nranks: 2
current endpoint: 127.0.0.1:29282
input: [1 1]
allreduce: [3 3]
c_allreduce: [3 3]
c_allgather: [1 1 2 2]
c_reducescatter: [2 2]
c_broadcast: [1 1]

local_rank: 1
nranks: 2
current endpoint: 127.0.0.1:57258
input: [1 1]
allreduce: [3 3]
c_allreduce: [3 3]
c_allgather: [1 1 2 2]
c_reducescatter: [4 4]
c_broadcast: [1 1]
```
