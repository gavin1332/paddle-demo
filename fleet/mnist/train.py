from __future__ import print_function
import os
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker

import model
import utils


def init_dist_env():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)


def distributed_optimize(loss):
    strategy = DistributedStrategy()
    strategy.fuse_all_reduce_ops = True
    strategy.nccl_comm_num = 2 
    strategy.fuse_elewise_add_act_ops=True
    strategy.fuse_bn_act_ops = True

    optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    optimizer.minimize(loss, fluid.default_startup_program())


def create_executor():
    place = fluid.CUDAPlace(int(os.environ['FLAGS_selected_gpus']))
    exe = fluid.Executor(place)
    return exe


def train(train_prog, start_prog, exe, feed, fetch):
    train_prog = fleet.main_program
    train_loader = utils.create_dataloader(paddle.dataset.mnist.train(), feed, False)
    exe.run(start_prog)
    for idx, sample in enumerate(train_loader()):
        ret = exe.run(train_prog, feed=sample, fetch_list=fetch)
        if idx % 100 == 0:
            print('[TRAIN] step=%d loss=%f' % (idx, ret[0][0]))


def test(test_prog, exe, feed, fetch):
    test_loader = utils.create_dataloader(paddle.dataset.mnist.test(), feed, True)
    acc_manager = fluid.metrics.Accuracy()
    for idx, sample in enumerate(test_loader()):
        ret = exe.run(test_prog, feed=sample, fetch_list=fetch)
        acc_manager.update(value=ret[0], weight=utils.sample_batch(sample))
        if idx % 100 == 0:
            print('[TEST] step=%d accum_acc1=%.2f' % (idx, acc_manager.eval()))
    print('[TEST] local_acc1: %.2f' % acc_manager.eval())

    return acc_manager.value, acc_manager.weight


if __name__ == '__main__':
    init_dist_env()
    exe = create_executor()

    train_prog, start_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_prog, start_prog):
        feed, fetch = model.build_train_net()
        distributed_optimize(fetch[0])    
    train(train_prog, start_prog, exe, feed, fetch)

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog):
        feed, fetch = model.build_test_net()
    local_acc, local_weight = test(test_prog, exe, feed, fetch)

    dist_acc = utils.dist_eval_acc(exe, local_acc, local_weight)
    print('[TEST] global_acc1: %.2f' % dist_acc)
