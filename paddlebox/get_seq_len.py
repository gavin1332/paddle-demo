from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
ones = layers.ones_like(slot)
float_ones = layers.cast(ones, dtype='float32')
value = layers.sequence_pool(float_ones, pool_type='sum')

feed_list = {
    'slot': fluid.create_lod_tensor(np.array([[0],[1],[2],
                                              [3],[4]], dtype='int64'), [[3, 2]], fluid.CPUPlace())
    }
fetch_list = [value]
exe = fluid.Executor(fluid.CPUPlace())
result = exe.run(fluid.default_main_program(), feed=feed_list, fetch_list=fetch_list)
print('sequence length:', result)
