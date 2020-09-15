#!/bin/bash

UPLOAD_DIR=_upload
echo "[INFO] generating data into folder $UPLOAD_DIR ..." >&2

python/bin/python rm_key.py $UPLOAD_DIR

if [ $? -ne 0 ]; then
    echo "[ERROR] fail to generate data" >&2
    exit 1
fi
echo "[INFO] finish generating data" >&2

OUTPUT_DIR="${mapred_output_dir}/_temporary/_${mapred_task_id}"
echo "[INFO] uploading data file to hdfs $OUTPUT_DIR/ ..." >&2
$HADOOP_HOME/bin/hadoop fs \
    -Dhadoop.job.ugi=valyria,valyria \
    -Dfs.default.name=afs://wudang.afs.baidu.com:9902 \
    -put $UPLOAD_DIR/* $OUTPUT_DIR/

if [ $? -ne 0 ]; then
    echo "[ERROR] fail to upload data file" >&2
    exit 1
fi
echo "[INFO] finish uploading data file" >&2
