import os
import paddle.fluid as fluid
import numpy as np
#from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy  # new line 1 
#from paddle.fluid.incubate.fleet.base import role_maker # new line 2
#
#role = role_maker.PaddleCloudRoleMaker(is_collective=True) # new line 3
#fleet.init(role) # new line 4

x = fluid.data(name='x', shape=[-1, 2], dtype='float32')
label = fluid.data(name='label', shape=[-1, 1], dtype='float32')
y = fluid.layers.fc(x, size=1, param_attr=fluid.initializer.Constant(1.0))
fluid.layers.Print(y)

#v1 = fluid.layers.collective._c_allgather(y, fleet.worker_num(), use_calc_stream=True)
#v2 = fluid.layers.collective._c_allreduce(y, use_calc_stream=True)
#fluid.layers.Print(v1)
#fluid.layers.Print(v2)

cost = fluid.layers.square_error_cost(y, label)
loss = fluid.layers.reduce_sum(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.0)
#strategy = DistributedStrategy()
#strategy.mode = "collective"
#strategy.collective_mode = "grad_allreduce"
#optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy) # new line 5
optimizer.minimize(loss, fluid.default_startup_program())

place = fluid.CUDAPlace(0) # to be commented line 1
#place = fluid.CUDAPlace(int(os.environ['FLAGS_selected_gpus'])) # uncomment line 1
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_prog = fluid.default_main_program() # to be commented line 2
#train_prog = fleet.main_program # uncomment line 2
x_data = np.ones(shape=[1, 2], dtype=np.float32)
label_data = np.ones(shape=[1, 1], dtype=np.float32)
out = exe.run(train_prog,
    feed={'x': x_data, 'label': label_data},
    fetch_list=[loss.name])
