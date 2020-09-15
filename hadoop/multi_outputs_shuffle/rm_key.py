from __future__ import print_function
import sys
import os

if len(sys.argv) != 2:
    print('Usage: %s %s tmp_dir' % (sys.executable, sys.argv[0]))
    sys.exit()

upload_dir = sys.argv[1]

time_last = None
part_last = None
file = None
cnt = 0
for line in sys.stdin:
    if cnt % 1000 == 0:
        print("%d line parsed" % cnt, file=sys.stderr)

    idx = line.index('\t')
    time, part = line[0:idx].split('/')

    if time != time_last:
        os.makedirs(upload_dir + '/' + time)
        time_last = time
        part_last = None

    if part != part_last:
        if file:
            file.close()
        filepath = '%s/%s/%s' % (upload_dir, time, part)
        file = open(filepath, 'w')
        part_last = part

    print(line[idx+1:], end='', file=file)
    cnt += 1

if file:
    file.close()
