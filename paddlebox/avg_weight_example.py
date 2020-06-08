# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import paddle.fluid.core as core

OpRole = core.op_proto_and_checker_maker.OpRole
def _is_backward_op(op, op_role_key):
    return op_role_key in op.attr_names and \
        int(op.all_attrs()[op_role_key]) & int(OpRole.Backward)


avgw_list = []
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

    decay_var = layers.fill_constant(shape=[1], value=0.9, dtype='float32')
    rev_decay_var = layers.fill_constant(shape=[1], value=0.1, dtype='float32')

    block = main_program.global_block()
    op_maker = core.op_proto_and_checker_maker
    op_role_key = op_maker.kOpRoleAttrName() # "op_role"
    op_role_var_key = op_maker.kOpRoleVarAttrName() # "op_role_var"
    param2avg = []
    for idx, op in list(enumerate(block.ops)):
        if _is_backward_op(op, op_role_key) and op_role_var_key in op.attr_names:
            op_role_var = op.all_attrs()[op_role_var_key]
            if len(op_role_var) == 0:
                continue
            assert len(op_role_var) % 2 == 0
            for i in range(0, len(op_role_var), 2):
                param = block.vars[op_role_var[i]]
                avg_var = layers.create_global_var(
                    name = param.name + "@avg",
                    shape = param.shape,
                    value = 1.0,
                    dtype = 'float32',
                    persistable =True)
                avgw_list.append(avg_var)

                tmp0 = layers.elementwise_mul(avg_var, decay_var)
                tmp1 = layers.elementwise_mul(param, rev_decay_var)
                block.append_op(
                    type='elementwise_add',
                    inputs={'X': tmp0,
                            'Y': tmp1},
                    outputs={'Out': avg_var},
                    stop_gradient=True) 

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

# 每个Pass前拷贝parameter到average weight
for avgw in avgw_list:
    param_name = avgw.name.split('@')[0]
    print("copy param: %s" % (param_name))
    dst_tensor = fluid.global_scope().find_var(avgw.name).get_tensor()
    dst_tensor.set(fluid.global_scope().find_var(param_name).get_tensor(), fluid.CPUPlace())

# 执行训练（模拟10个minibatch），指定输入，获取输出
for i in range(10):
    result = exe.run(main_program, feed=feed_list, fetch_list=fetch_list)
    print('avg_cost:', result)
