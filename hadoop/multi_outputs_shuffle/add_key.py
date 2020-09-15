from __future__ import print_function
import sys
import datetime
import random
import time
import itertools

if len(sys.argv) != 3:
    print('Usage: %s %s hour_start hour_last' % (sys.executable, sys.argv[0]))
    sys.exit()

hour_start = int(sys.argv[1])
hour_last = int(sys.argv[2])
num5m = (hour_last - hour_start + 1) * 60 // 5

time_start = datetime.datetime(2000, 1, 1, hour_start, 0)
times = [(time_start + delta).strftime('%H%M') for delta in [datetime.timedelta(minutes=md) for md in range(0, 5*num5m, 5)]]

part_range = 300
keys = ['%s/part_%d' % (pair[0], pair[1]) for pair in itertools.product(times, range(part_range))]

random.seed(int(time.time() * 1000))

idx = 0
random.shuffle(keys)
for line in sys.stdin.readlines():
    if idx >= len(keys):
        random.shuffle(keys)
        idx = 0

    print('%s\t%s' % (keys[idx], line), end='')
    idx += 1
