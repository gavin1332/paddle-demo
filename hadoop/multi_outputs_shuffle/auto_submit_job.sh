#!/bin/bash

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: sh $0 date_start [0_based_section_index_start] [date_last] [0_based_section_index_last]"
    echo
    echo "Description:"
    echo
    echo "    We have to specify a hour section index to trigger the automatical shuffling jobs."
    echo "    And these sequential jobs start with \"date_start\" and end with \"date_last\"."
    echo
    echo "    The total 24 hours are split to 5 hour sections with their index mapping are:"
    echo
    echo "    |hour section|index|"
    echo "    |------------|-----|"
    echo "    |  [0, 3]    |  0  |"
    echo "    |  [4, 7]    |  1  |"
    echo "    |  [8, 12]   |  2  |"
    echo "    |  [13, 18]  |  3  |"
    echo "    |  [19, 23]  |  4  |"
    echo

    exit
fi

source ./job.conf

SEC_LEFT=(0 4 8 13 19)
SEC_RIGHT=(3 7 12 18 23)

DATE=$1
SEC_IDX_START=0
if [[ $# -ge 2 ]]; then
    SEC_IDX_START=$2
fi
DATE_LAST=20991231
if [[ $# -ge 3 ]]; then
    DATE_LAST=$3
fi
SEC_IDX_LAST=${#SEC_LEFT[@]}-1
if [[ $# -ge 4 ]]; then
    SEC_IDX_LAST=$4
fi
echo "DATE: $DATE"
echo "SEC_IDX_START: $SEC_IDX_START"
echo "DATE_LAST: $DATE_LAST"
echo "SEC_IDX_LAST: $SEC_IDX_LAST"


while [ $DATE -le $DATE_LAST ]; do
    for ((i=SEC_IDX_START;i<=SEC_IDX_LAST;++i)); do
        echo "Current date: $DATE"

        HOUR_START=${SEC_LEFT[i]}
        HOUR_LAST=${SEC_RIGHT[i]}

        echo "Current hour section: [$HOUR_START, $HOUR_LAST]"

        echo "Start waiting for data ready ..."
        sh wait_for_data_ready.sh $DATE $HOUR_START $HOUR_LAST
        if [ $? -ne 0 ]; then
            echo "[ERROR] Fail to check original data status. Try the next section $i"
            continue
        fi

        echo "Start distcp ..."
        sh distcp.sh $DATE $HOUR_START $HOUR_LAST
        if [ $? -ne 0 ]; then
            echo "[ERROR] Fail to distcp original data. Try the next section $i"
            continue
        fi

        echo "Start shuffling mapreduce job ..."
        sh submit_job.sh $DATE $HOUR_START $HOUR_LAST
        if [ $? -ne 0 ]; then
            echo "[ERROR] Fail to run mr job. Try the next section $i"
            continue
        fi

        echo "Start moving final data ..."
        $HADOOP fs -Dhadoop.job.ugi=$TARGET_PATH_UGI -mkdir $TARGET_PATH_ROOT/$DATE
        $HADOOP fs -Dhadoop.job.ugi=$TARGET_PATH_UGI -mv $MR_PATH_ROOT/$DATE/* $TARGET_PATH_ROOT/$DATE
    done

    DATE=$(date -d"$DATE +1day" +%Y%m%d)
    SEC_IDX_START=0
    SEC_IDX_LAST=${#SEC_LEFT[@]}-1
done
