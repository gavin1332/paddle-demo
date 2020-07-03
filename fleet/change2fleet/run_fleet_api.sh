CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --selected_gpus=0,1 --log_dir=log fleet_api.py
