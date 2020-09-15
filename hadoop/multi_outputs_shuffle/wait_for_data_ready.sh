#!/bin/bash

if [[ $# != 3 ]]; then
    echo "usage: $0 date hour_start hour_last"
    exit 0
fi

source ./job.conf

DATE=$1
HOUR_START=$2
HOUR_LAST=$3

HOUR=`printf "%02d" $HOUR_LAST`
TARGET=$ORIGIN_PATH_ROOT/$DATE/${HOUR}55/to.hadoop.done

TIME_START=$(date +%s)
while true; do
    $HADOOP fs -Dhadoop.job.ugi=$ORIGIN_PATH_UGI -ls $TARGET 2> /dev/null 
    if [ $? -eq 0 ]; then
        break
    fi

    echo "Dataset not ready, sleep 60s to check: $TARGET"
    sleep 60

    ((TIME_WAIT=$(date +%s)-$TIME_START))
    if [[ $TIME_WAIT -ge $TIMEOUT_FOR_MONITOR ]]; then
        exit 1
    fi
done
