import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

# Program to generate parameter
# The original 'weight' is filled value 1 with shape (4, 8)
ones = np.ones((4, 8)).astype('float32')
main_prog = fluid.Program()
start_prog = fluid.Program()
with fluid.program_guard(main_prog, start_prog):
    input = fluid.data('input', shape=[-1, 4], dtype='float32')
    output = layers.fc(input, 8,
        param_attr=fluid.ParamAttr(name='weight',
            initializer=fluid.initializer.NumpyArrayInitializer(
                ones)))

exe = fluid.Executor(fluid.CPUPlace())
# initialize all parameters
exe.run(start_prog)

# simulate saving model
fluid.io.save_persistables(exe, dirname="old", main_program=main_prog)

#############################################################################
# The following section illustrates what user should do to adjust parameter #
#############################################################################

# The target 'weight' is the concatenation of original 'weight' and a
# supplement weight filled 0 of shape (4, 8)
zeros = np.zeros((4, 8)).astype('float32')
main_prog = fluid.Program()
start_prog = fluid.Program()
with fluid.program_guard(main_prog, start_prog):
    w_part1 = layers.create_tensor(dtype='float32')
    layers.load(w_part1, file_path='old/weight')
    w_part2 = layers.assign(zeros)

    new_w = layers.concat([w_part1, w_part2], axis=0)
    main_prog.current_block().append_op(
        type='save',
        inputs={'X': [new_w]},
        outputs={},
        attrs={'file_path': 'new/weight'})

exe = fluid.Executor(fluid.CPUPlace())
exe.run(start_prog)
ret = exe.run(main_prog, fetch_list=[new_w.name])
target = np.concatenate((ones, zeros), axis=0)
assert np.array_equal(ret[0], target)
