import paddle.fluid as fluid
import numpy as np

a = fluid.data(name="a", shape=[1], dtype='float32')
b = fluid.data(name="b", shape=[1], dtype='float32')

result = fluid.layers.elementwise_add(a, b)

exe = fluid.Executor(fluid.CPUPlace())

x = np.array([5]).astype("float32")
y = np.array([7]).astype("float32")
outs = exe.run(fluid.default_main_program(), feed={'a':x,'b':y}, fetch_list=[result])
print(outs) #[array([12.], dtype=float32)]
