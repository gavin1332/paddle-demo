import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet


BATCH_SIZE = 16

def create_dataloader(generator, feed, place, is_test, is_distributed):
    def _dist_wrapper(generator):
        def _wrapper():
            rank = fleet.worker_index()
            nranks = fleet.worker_num()
            for idx, sample in enumerate(generator()):
                if idx % nranks == rank:
                    yield sample
        return _wrapper

    if is_distributed:
        generator = _dist_wrapper(generator)

    drop_last = False if is_test else True
    loader = fluid.io.DataLoader.from_generator(feed_list=feed, capacity=16)
    loader.set_sample_generator(generator, batch_size=BATCH_SIZE,
            drop_last=drop_last, places=[place])
    return loader


def dist_eval_acc(exe, local_acc, local_weight):
    prog = fluid.Program()
    with fluid.program_guard(prog):
        acc = fluid.layers.data(name='acc', shape=[1], dtype='float32')
        weight = fluid.layers.data(name='weight', shape=[1], dtype='float32')
        dist_acc = fluid.layers.collective._c_allreduce(acc, reduce_type='sum', use_calc_stream=True)
        dist_weight = fluid.layers.collective._c_allreduce(weight, reduce_type='sum', use_calc_stream=True)
    acc_sum, weight_sum = exe.run(prog, feed={'acc': local_acc, 'weight': local_weight}, fetch_list=[dist_acc, dist_weight]) 
    return acc_sum / weight_sum


def sample_batch(sample):
    tensor = list(sample[0].values())[0]
    assert(isinstance(tensor, fluid.LoDTensor))
    return tensor.shape()[0]

