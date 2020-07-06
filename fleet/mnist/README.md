# 简介
本示例基于MNIST数据集，展现如何使用Fleet API实现Paddle分布式训练。

我们在文件`train.py`中通过`args.distributed`配置项显式呈现了单机单卡代码如何使用Fleet API合入分布式训练能力。并且，**为了更加清晰地展现单卡训练和分布式训练在配置上的关键区别，避免分布式的配置分散不易阅读，我们将`args.distributed`配置项限制仅在主函数`main`里使用，以便于读者快速借鉴**。

即在`main`函数中，与分布式训练相关的代码要么在`if args.distributed`条件下执行，要么代入`args.distributed`参数。而其他可以在单卡训练和分布式训练中共享的代码均与此参数无关。

本示例需要基于paddle 1.8及以上版本运行，主要是分布式预测部分依赖1.8版本的特性。如果移除分布式预测代码，分布式训练部分的代码可以在更早版本的Paddle上运行。

# 文件结构

我们按照常见的方式创建了四个文件：

* `model.py`: 模型网络结构描述，含训练网络和测试网络
* `utils.py`: 工具函数
* `train.py`: 组建训练和评估流程的主函数
* `dist_run.sh`: 用于便捷启动分布式训练的脚本

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

假设有两台主机，ip地址分别为192.168.0.1和192.168.0.2，分别在两台主机上执行下述命令即可：

在ip地址为192.168.0.1的主机上：

``` code::bash
current_node_ip=192.168.0.1
cluster_node_ips=192.168.0.1,192.168.0.2

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

或直接执行

```
sh multinodes/node1_run.sh
```

在ip地址为192.168.0.2的主机上：

``` code::bash
current_node_ip=192.168.0.2
cluster_node_ips=192.168.0.1,192.168.0.2

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

或直接执行

```
sh multinodes/node2_run.sh
```

## 基于PaddleCloud执行多机多卡

待补充
