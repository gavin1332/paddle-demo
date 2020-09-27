cluster_name="v100-32-0-cluster"
group_name="dltp-0-yq01-k8s-gpu-v100-8"

k8s_wall_time="99:00:00"
k8s_priority="high"
k8s_trainers="1"
gpus_per_node="8"

job_version="paddle-fluid-v1.8.3"

job_name="mnist_n${k8s_trainers}"

distributed_conf="1 "
if [ ${k8s_trainers} -gt 1 ]
then
    distributed_conf="0 --distribute-job-type NCCL2 "
fi

upload_files="before_hook.sh end_hook.sh ../*.py ../dataset"

# 启动命令
start_cmd="python -m paddle.distributed.launch \
                  --use_paddlecloud \
                  --selected_gpus=0,1 \
                  --log_dir=mylog \
                  train.py --distributed"

paddlecloud job train \
    --job-name ${job_name} \
    --group-name ${group_name} \
    --cluster-name ${cluster_name} \
    --job-conf job.cfg \
    --start-cmd "${start_cmd}" \
    --files ${upload_files} \
    --job-version ${job_version}  \
    --k8s-gpu-cards $gpus_per_node \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-cpu-cores 35 \
    --k8s-trainers ${k8s_trainers} \
    --k8s-priority ${k8s_priority} \
    --is-standalone ${distributed_conf} 
