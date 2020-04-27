from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers

####################
# original program #
####################
main_prog = fluid.Program()
start_prog = fluid.Program()
with fluid.program_guard(main_prog, start_prog):
    slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
    label = fluid.data('label', [-1, 1])
    emb = layers.embedding(slot, [4, 12], param_attr=fluid.ParamAttr(name="emb"))
    pool = layers.sequence_pool(emb, 'sum')
    fc = layers.fc(pool, 12, act='relu')
    logit = layers.fc(fc, 1)
    loss = layers.sigmoid_cross_entropy_with_logits(logit, label)

exe = fluid.Executor(fluid.CUDAPlace(0))
# if no GPU is available, use statement below:
#exe = fluid.Executor(fluid.CPUPlace())

# initialize all parameters
exe.run(start_prog)

fluid.io.save_persistables(exe, dirname="model", main_program=main_prog)

# show all parameters in the original model
param_names = {var.name for var in main_prog.list_vars() if var.persistable}
print(param_names)

############################################
# create new program with additional slots #
############################################
main_prog_new = fluid.Program()
start_prog_new = fluid.Program()
with fluid.program_guard(main_prog_new, start_prog_new):
    slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
    label = fluid.data('label', [-1, 1])
    emb = layers.embedding(slot, [4, 12], param_attr=fluid.ParamAttr(name="emb"))
    pool = layers.sequence_pool(emb, 'sum')
    fc = layers.fc(pool, 12)

    slot_new = fluid.data('slot_new', [-1, 1], dtype='int64', lod_level=1)
    emb_new = layers.embedding(slot_new, [4, 12], param_attr=fluid.ParamAttr(name="emb"))
    pool_new = layers.sequence_pool(emb_new, 'sum')
    fc_new = layers.fc(pool_new, 12)

    fc_sum = fc + fc_new
    fc_relu = layers.relu(fc_sum)
    logit = layers.fc(fc_relu, 1)
    loss = layers.sigmoid_cross_entropy_with_logits(logit, label)

# initialize all parameters
exe.run(start_prog_new)
# override parameters listed in param_names
# reference: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/io_cn/load_vars_cn.html#load-vars
fluid.io.load_vars(exe, dirname='model', main_program=main_prog_new, predicate=(lambda v: v.name in param_names))
