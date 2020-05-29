# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

# 自定义Main Program和Start Program
main_program = fluid.Program()
start_program = fluid.Program()
with fluid.program_guard(main_program, start_program):
    # 组网
    slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
    label = fluid.data('label', [-1, 1])
    emb = layers.embedding(slot, [5, 12])
    pool = layers.sequence_pool(emb, 'sum')
    logit = layers.fc(pool, 1)
    loss = layers.sigmoid_cross_entropy_with_logits(logit, label)
    avg_cost = layers.mean(loss)
    
    # 定义优化器
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)

# 执行器声明
exe = fluid.Executor(fluid.CPUPlace())
# 执行初始化
exe.run(start_program)

# 定义输入
feed_list = {
    'slot': fluid.create_lod_tensor(np.array([[0],[1],[2],
                                              [3],[4]], dtype='int64'), [[3, 2]], fluid.CPUPlace()),
    'label': np.array([[1], [0]], dtype='float32')
    }

fetch_list = [avg_cost]
# 执行训练，指定输入，获取输出
result = exe.run(main_program, feed=feed_list, fetch_list=fetch_list)
print('avg_cost:', result) #avg_cost: [array([0.61414886], dtype=float32)]
