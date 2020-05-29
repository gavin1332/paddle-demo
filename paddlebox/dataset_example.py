# -*- coding: UTF-8 -*-
# @Author: xujiaqi01@baidu.com
import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid
 
# 假如我们有两个数据文件如下：
with open('a.txt', 'w') as f:
    f.write('1 1 2 8 9 4 15 25 35 45 1 21\n')
    f.write('1 2 2 8 10 4 16 26 36 46 1 22\n')
    f.write('1 3 2 9 10 4 17 27 37 47 1 23')
with open('b.txt', 'w') as f:
    f.write('1 4 2 8 11 4 15 25 36 45 1 24\n')
    f.write('1 5 2 10 12 4 26 36 46 56 1 85\n')
    f.write('1 6 2 12 13 4 17 27 37 67 1 86\n')
    f.write('1 7 2 9 11 4 18 28 38 48 1 87')
 
# 定义网络，本示例只定义了数据层
slots = ['slot1','slot2','slot3','slot4']
slots_vars = []
for slot in slots:
    var = fluid.layers.data(name=slot, shape=[1], dtype='int64', lod_level=1)
    slots_vars.append(var)
    fluid.layers.Print(var)
# 创建Dataset，并配置
dataset = fluid.DatasetFactory().create_dataset()
dataset.set_batch_size(7)
dataset.set_thread(1)
dataset.set_filelist(['a.txt', 'b.txt'])
dataset.set_pipe_command('cat')
dataset.set_use_var(slots_vars)
# 跑训练
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
exe.train_from_dataset(fluid.default_main_program(), dataset)
