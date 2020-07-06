# 简介
本示例基于MNIST数据集，展现如何使用Fleet API实现Paddle分布式训练。

# 文件结构

我们按照常见的方式创建了四个文件：

* model.py: 模型网络结构描述，含训练网络和测试网络
* utils.py: 工具函数
* train.py: 组建训练和评估流程的主函数
* dist_run.sh: 用于便捷启动分布式训练的脚本

其中，`train.py`文件中通过`args.distributed`配置项清晰地呈现了单机单卡代码如何便捷地使用Fleet API合入分布式训练能力。并且，仅有主函数会使用`args.distributed`配置项，从而单卡训练配置和多卡分布式训练配置的关键区别能够更加清晰地展现，便于读者快速借鉴。

# 执行方法

## 单机单卡

``` code::bash
python train.py
```

## 单机多卡

``` code::bash
sh dist_run.sh
```

## 多机多卡

假设有两台主机，ip地址分别为192.168.0.2和192.168.0.3，分别在两台主机上执行下述命令即可：

在ip地址为192.168.0.2的主机上：

``` code::bash
current_node_ip=192.168.0.2
cluster_node_ips=192.168.0.2,192.168.0.3

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

在ip地址为192.168.0.3的主机上：

``` code::bash
# on node 192.168.0.3
current_node_ip=192.168.0.3
cluster_node_ips=192.168.0.2,192.168.0.3

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

## 基于PaddleCloud执行多机多卡

待补充
