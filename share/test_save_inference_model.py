from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.transpiler.details.program_utils import program_to_code

main_program = fluid.Program()
start_program = fluid.Program()
with fluid.program_guard(main_program, start_program):
    slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
    label = fluid.data('label', [-1, 1])
    emb = layers.embedding(slot, [4, 12])
    pool = layers.sequence_pool(emb, 'sum')
    fc = layers.fc(pool, 12, act='relu')
    logit = layers.fc(fc, 1)
    loss = layers.sigmoid_cross_entropy_with_logits(logit, label)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(start_program)

fluid.io.save_inference_model(dirname="model/",
                              feeded_var_names=['slot'],
                              target_vars=[logit],
                              executor=exe,
                              main_program=main_program)
