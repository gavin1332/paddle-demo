# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core

OpRole = core.op_proto_and_checker_maker.OpRole
def _is_backward_op(op, op_role_key):
    return op_role_key in op.attr_names and \
        int(op.all_attrs()[op_role_key]) & int(OpRole.Backward)

# 自定义Main Program和Start Program
main_program = fluid.Program()
start_program = fluid.Program()
init_avgw_program = fluid.Program()
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
    start_block = start_program.global_block()
    init_avgw_block = init_avgw_program.global_block()
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
    
                # startup program
                avg_var = start_block.create_var(name=param.name + "@avg",
                                                 shape=param.shape,
                                                 persistable=True,
                                                 dtype=param.dtype)
                start_block.append_op(type='assign',
                                      inputs={'X': [param]},
                                      outputs={'Out': [avg_var]})

                # init average weight program
                from_var = init_avgw_block.create_var(name=param.name,
                                                      shape=param.shape,
                                                      persistable=True,
                                                      dtype=param.dtype)
                to_var = init_avgw_block.create_var(name=avg_var.name,
                                                    shape=avg_var.shape,
                                                    persistable=True,
                                                    dtype=avg_var.dtype)
                init_avgw_block.append_op(type='assign',
                                          inputs={'X': [from_var]},
                                          outputs={'Out': [to_var]})

                # main program
                avg_var = block.create_var(
                        name=avg_var.name,
                        shape=avg_var.shape,
                        persistable=True,
                        dtype=avg_var.dtype,
                        stop_gradient=True)
                tmp0 = layers.elementwise_mul(avg_var, decay_var)
                tmp1 = layers.elementwise_mul(param, rev_decay_var)
                block.append_op(
                    type='elementwise_add',
                    inputs={'X': tmp0,
                            'Y': tmp1},
                    outputs={'Out': avg_var},
                    stop_gradient=True) 

# 打印网络
with open('start.program', 'w') as fout:
    print(start_program, file=fout)
with open('init_avgw.program', 'w') as fout:
    print(init_avgw_program, file=fout)
with open('main.program', 'w') as fout:
    print(main_program, file=fout)

# 执行器声明
exe = fluid.Executor(fluid.CPUPlace())
# 执行初始化
exe.run(start_program)

# 每个pass前拷贝weight到average_weight
exe.run(init_avgw_program)

# 定义输入
feed_list = {
    'slot': fluid.create_lod_tensor(np.array([[0],[1],[2],
                                              [3],[4]], dtype='int64'), [[3, 2]], fluid.CPUPlace()),
    'label': np.array([[1], [0]], dtype='float32')
    }

fetch_list = [avg_cost]

# 执行训练（模拟10个minibatch），指定输入，获取输出
for i in range(10):
    result = exe.run(main_program, feed=feed_list, fetch_list=fetch_list)
    print('avg_cost:', result)
