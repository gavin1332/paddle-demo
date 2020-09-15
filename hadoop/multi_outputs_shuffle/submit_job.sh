#!/bin/bash

set -o pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 date hour_start hour_last"
    exit 0
fi

source ./job.conf

DATE=$1
HOUR_START=$2
HOUR_LAST=$3

TIME_PATTERN=`printf "%02d*" $HOUR_START`
INPUT_PATH=$DISTCP_PATH_ROOT/$DATE/$TIME_PATTERN/part_*
for ((i=HOUR_START+1; i<=HOUR_LAST; ++i)); do
    TIME_PATTERN=`printf "%02d*" $i`
    INPUT_PATH+=",$DISTCP_PATH_ROOT/$DATE/$TIME_PATTERN/part_*"
done
OUTPUT_PATH=$MR_PATH_ROOT/$DATE

echo "INPUT: $INPUT_PATH"
echo "OUTPUT: $OUTPUT_PATH"
$HADOOP fs -rmr $OUTPUT_PATH 2> /dev/null

MAPPER="python/bin/python add_key.py $HOUR_START $HOUR_LAST"
REDUCER="sh reduce.sh"

echo "MAPPER: $MAPPER"
echo "REDUCER: $REDUCER"

TASK_NAME=${JOB_NAME}_${USERNAME}_${DATE}_${HOUR_START}_${HOUR_LAST}
echo "Start submiting mapreduce task $TASK_NAME ..."

$HADOOP streaming \
    -D mapred.job.priority=$PRIORITY \
    -D mapred.job.map.capacity=$MAP_CAPACITY \
    -D mapred.job.reduce.capacity=$REDUCE_CAPACITY \
    -D mapred.map.tasks=$MAP_TASK \
    -D mapred.reduce.tasks=$REDUCE_TASK \
    -D mapred.textoutputformat.ignoreseparator=true \
    -D mapred.min.split.size=41943040 \
    -D mapred.job.name=$TASK_NAME \
    -D stream.memory.limit=500 \
    -input $INPUT_PATH \
    -output $OUTPUT_PATH \
    -mapper "$MAPPER" \
    -reducer "$REDUCER" \
    -file ./add_key.py \
    -file ./rm_key.py \
    -file ./reduce.sh \
    -cacheArchive afs://wudang.afs.baidu.com:9902/python/path/python27-gcc48.tgz#python

if [ $? -ne 0 ]; then
    echo "[ERROR] hadoop job failed"
    exit 1
fi

echo "[INFO] Cleaning output path ..."
$HADOOP fs -rmr $OUTPUT_PATH/_temporary &> /dev/null
$HADOOP fs -rm $OUTPUT_PATH/part-* &> /dev/null

echo "[INFO] touch to.hadoop.done ..."
$HADOOP fs -Dhadoop.job.ugi=$TARGET_PATH_UGI -ls $OUTPUT_PATH \
        | awk 'NF==8{print $8}' | awk -F"/" '{print $NF}' \
        | xargs -i -n 1 $HADOOP fs -Dhadoop.job.ugi=$TARGET_PATH_UGI -touchz $OUTPUT_PATH/{}/to.hadoop.done

echo "[INFO] Job done"

