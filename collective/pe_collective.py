import os
import time
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

# PE training
main_prog = fluid.Program()
start_prog = fluid.Program()
with fluid.program_guard(main_prog, start_prog):
    x = fluid.data(name='x', shape=[-1, 2], dtype='float32')
    label = fluid.data(name='label', shape=[-1, 1], dtype='float32')
    y = fluid.layers.fc(x, size=1, param_attr=fluid.initializer.Constant(1.0))

    cost = fluid.layers.square_error_cost(y, label)
    loss = fluid.layers.reduce_sum(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.0)
strategy = DistributedStrategy()
optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
optimizer.minimize(loss, start_prog)

place = fluid.CUDAPlace(int(os.environ['FLAGS_selected_gpus']))
exe = fluid.Executor(place)
exe.run(start_prog)

train_prog = fleet.main_program
x_data = np.ones(shape=[1, 2], dtype=np.float32)
label_data = np.ones(shape=[1, 1], dtype=np.float32)
out = exe.run(train_prog,
    feed={'x': x_data, 'label': label_data},
    fetch_list=[loss.name])

# EXE testing
test_prog = fluid.Program()
with fluid.program_guard(test_prog):
    y = fluid.data(name='y', shape=[-1, 1], dtype='float32')
    v1 = fluid.layers.collective._c_allgather(y, fleet.worker_num(), use_calc_stream=True)
    v2 = fluid.layers.collective._c_allreduce(y, use_calc_stream=True)
y_data = np.ones(shape=[1, 1], dtype=np.float32)
v1, v2 = exe.run(test_prog, feed={'y': y_data}, fetch_list=[v1.name, v2.name])

if role.worker_index() == 1:
    time.sleep(1)
    print ""

print 'rank:', role.worker_index()
print v1, v2
