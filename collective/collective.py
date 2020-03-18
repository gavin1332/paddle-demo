#!/usr/bin/env python

import time
from paddle import fluid
from paddle.fluid.layers import collective

if __name__ == "__main__":
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        s = fluid.dygraph.parallel.prepare_context()
        print(s.local_rank, s.nranks, s.current_endpoint)

        tensor = fluid.layers.zeros([2], 'int32')
        if s.local_rank == 0:
            tensor += 1
        if s.local_rank == 1:
            tensor += 2
        allr = collective._allreduce(tensor)
        callr = collective._c_allreduce(tensor, use_calc_stream=True)
        callg = collective._c_allgather(tensor, s.nranks, use_calc_stream=True)
        creds = collective._c_reducescatter(callg, s.nranks, use_calc_stream=True)
        cbroad = collective._c_broadcast(tensor, use_calc_stream=True)
        if s.local_rank == 0:
            print("")
            print('local_rank:', s.local_rank, 'input:', tensor.numpy())
            print('allreduce:', allr.numpy())
            print('c_allreduce:', callr.numpy())
            print('c_allgather:', callg.numpy())
            print('c_reducescatter:', creds.numpy())
            print('c_broadcast:', cbroad.numpy())
        else:
            time.sleep(2)
            print("")
            print('local_rank:', s.local_rank, 'input:', tensor.numpy())
            print('allreduce:', allr.numpy())
            print('c_allreduce:', callr.numpy())
            print('c_allgather:', callg.numpy())
            print('c_reducescatter:', creds.numpy())
            print('c_broadcast:', cbroad.numpy())

