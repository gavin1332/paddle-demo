# 简介
本示例基于MNIST数据集，展现如何使用Fleet API实现Paddle分布式训练。

# 文件结构

我们按照常见的方式创建了四个文件：

* model.py: 模型网络结构描述，含训练网络和测试网络
* utils.py: 工具函数
* train.py: 组建训练和评估流程的主函数
* dist_run.sh: 用于便捷启动分布式训练的脚本

其中，我们在`train.py`文件中通过`args.distributed`配置项清晰地呈现了单机单卡代码如何便捷地使用Fleet API加入分布式训练能力。

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

待补充

## 基于PaddleCloud执行多机多卡

待补充
