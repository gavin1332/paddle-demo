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
trainers_endpoints: 127.0.0.1:15159,127.0.0.1:46191 , node_id: 0 , current_node_ip: 127.0.0.1 , num_nodes: 1 , node_ips: ['127.0.0.1'] , nranks: 2
I0318 16:36:50.957653 113052 nccl_context.cc:127] init nccl context nranks: 2 local rank: 0 gpu id: 0
I0318 16:36:50.957679 113053 nccl_context.cc:127] init nccl context nranks: 2 local rank: 1 gpu id: 1
W0318 16:36:52.364046 113053 device_context.cc:237] Please NOTE: device: 1, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0318 16:36:52.372650 113052 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.2
W0318 16:36:52.401707 113052 device_context.cc:245] device: 0, cuDNN Version: 7.4.
W0318 16:36:52.401739 113053 device_context.cc:245] device: 1, cuDNN Version: 7.4.
(1L, 2L, u'127.0.0.1:46191')
(0L, 2L, u'127.0.0.1:15159')

('local_rank:', 0L, 'input:', array([1, 1], dtype=int32))
('allreduce:', array([3, 3], dtype=int32))
('c_allreduce:', array([3, 3], dtype=int32))
('c_allgather:', array([1, 1, 2, 2], dtype=int32))
('c_reducescatter:', array([2, 2], dtype=int32))
('c_broadcast:', array([1, 1], dtype=int32))

('local_rank:', 1L, 'input:', array([1, 1], dtype=int32))
('allreduce:', array([3, 3], dtype=int32))
('c_allreduce:', array([3, 3], dtype=int32))
('c_allgather:', array([1, 1, 2, 2], dtype=int32))
('c_reducescatter:', array([4, 4], dtype=int32))
('c_broadcast:', array([1, 1], dtype=int32))
```
