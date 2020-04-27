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

# Method 1: Print structural program description
with open('main_program.1', 'w') as fout:
    print(main_program, file=fout)

# Method 2: Print compact structural program description
with open('main_program.2', 'w') as fout:
    program_to_code(main_program, fout=fout)

# Method 3: Print binary program data with a specified filename for visualization
with open('__model__', "wb") as fout:
    fout.write(main_program.desc.serialize_to_string())
