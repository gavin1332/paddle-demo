CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --selected_gpus=0,1 fleet_api.py
