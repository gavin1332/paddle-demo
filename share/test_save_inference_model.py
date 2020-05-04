from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers

main_program = fluid.Program()
start_program = fluid.Program()
with fluid.program_guard(main_program, start_program):
    slot = fluid.data('slot', [-1, 1], dtype='int64', lod_level=1)
    label = fluid.data('label', [-1, 1])
    emb = layers.embedding(slot, [4, 12])
    pool = layers.sequence_pool(emb, 'sum')
    fc = layers.fc(pool, 12, act='relu')
    fc = layers.scale(fc, scale=1.0)
    logit = layers.fc(fc, 1)
    loss = layers.sigmoid_cross_entropy_with_logits(logit, label)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(start_program)

for i in range(3):
    program = main_program.clone()
    with fluid.unique_name.guard():
        fluid.io.save_inference_model(dirname='model/',
                                      model_filename=None,
                                      params_filename=None,
                                      feeded_var_names=['slot'],
                                      target_vars=[logit],
                                      executor=exe,
                                      main_program=program)

last_var = None
for var in program.current_block().vars:
    last_var = var
assert(last_var == 'save_infer_model/scale_0.tmp_0')
