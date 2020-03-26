#!/usr/bin/env python

import time
from paddle import fluid
from paddle.fluid.layers import collective

def print_result(result):
    print("")
    for item in result:
      print '%s: %s' % (item[0], item[1])

if __name__ == "__main__":
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        s = fluid.dygraph.parallel.prepare_context()

        tensor = fluid.layers.zeros([2], 'int32')
        if s.local_rank == 0:
            tensor += 1
        if s.local_rank == 1:
            tensor += 2

        allr = collective._allreduce(tensor)
        # Since the following collective operators act in communication stream
        # asynchronously by default, we have to force them in calculation stream
        # (use_calc_stream=True) to keep data synchronous, or call sync_comm_stream
        # operator explicitly to synchronize data. However, sync_comm_stream has no
        # computing kernel that makes it failed in dygraph.
        callr = collective._c_allreduce(tensor, use_calc_stream=True)
        callg = collective._c_allgather(tensor, s.nranks, use_calc_stream=True)
        creds = collective._c_reducescatter(callg, s.nranks, use_calc_stream=True)
        cbroad = collective._c_broadcast(tensor, use_calc_stream=True)

        result = [('local_rank', s.local_rank),
                  ('nranks', s.nranks),
                  ('current endpoint', s.current_endpoint),
                  ('input', tensor.numpy()),
                  ('allreduce', allr.numpy()),
                  ('c_allreduce', callr.numpy()),
                  ('c_allgather', callg.numpy()),
                  ('c_reducescatter', creds.numpy()),
                  ('c_broadcast', cbroad.numpy())]

        if s.local_rank == 1:
            time.sleep(1)
        print_result(result)
