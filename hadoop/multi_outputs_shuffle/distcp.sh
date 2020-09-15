#!/bin/bash

JOB_UGI=XXX
JOB_TRACKER=XXX
JOB_QUEUE=XXX

if [[ $# != 3 ]]; then
    echo "usage: $0 date hour_start hour_last"
    exit 0
fi

source ./job.conf

DATE=$1
HOUR_START=$2
HOUR_LAST=$3

for ((i=$HOUR_START; i<=$HOUR_LAST; ++i)); do
    HOUR=`printf "%02d" $i`
    for ((j=0; j<60; j+=5)); do
        MINUTES=`printf "%02d" $j`
        SRC_PATH+="$ORIGIN_PATH_ROOT/$DATE/$HOUR$MINUTES "
    done
done
DST_PATH=$DISTCP_PATH_ROOT/$DATE

echo "SRC_PATH: $SRC_PATH"
echo "DST_PATH: $DST_PATH"

nmaps=$MAP_CAPACITY
priority=$PRIORITY

$HADOOP fs -rmr $DST_PATH
$HADOOP distcp \
    -Dfs.default.name=hdfs://yq01-wutai-hdfs.dmop.baidu.com:54310 \
    -Dhadoop.job.ugi=$JOB_UGI \
    -Dmapred.job.tracker=$JOB_TRACKER \
    -Dmapred.job.queue.name=$JOB_QUEUE \
    -Dmapred.job.name=distcp_${JOB_NAME}_${USERNAME} \
    -Dmapred.job.map.capacity=$nmaps \
    -Dmapred.job.priority=$priority \
    -su $ORIGIN_PATH_UGI \
    -du $TARGET_PATH_UGI \
    $SRC_PATH $DST_PATH

$HADOOP fs -Dhadoop.job.ugi=$TARGET_PATH_UGI -rmr $DST_PATH/_distcp_tmp_* 2> /dev/null
