# Transformer模型训练模板说明

本文档介绍如何使用`workflow_transformer.py`模板来训练Transformer模型进行股票预测。

## 功能概述

该模板实现了以下功能：
- 使用Qlib的TransformerModel进行股票预测模型训练
- 数据预处理和特征工程
- 模型训练、验证和测试
- 信号生成和分析
- 回测和投资组合分析

## 使用方法

1. 确保已安装Qlib及其依赖：
```bash
pip install pyqlib
```

2. 运行训练脚本：
```bash
python workflow_transformer.py
```

## 自定义选项

您可以根据自己的需求修改以下参数：

### 模型参数
在`model_config`中，您可以调整Transformer模型的超参数：
- `d_feat`: 特征数量（默认为20）
- `d_model`: 模型维度（默认为64）
- `nhead`: 注意力头数量（默认为2）
- `num_layers`: Transformer层数（默认为2）
- `dropout`: Dropout率（默认为0.0）
- `n_epochs`: 训练轮数（默认为100）
- `lr`: 学习率（默认为0.0001）
- `batch_size`: 批次大小（默认为8192）
- `GPU`: GPU ID（默认为0，负数表示使用CPU）

### 数据参数
在`data_handler_config`中，您可以调整数据处理相关参数：
- 时间范围：`start_time`、`end_time`、`fit_start_time`、`fit_end_time`
- 标的范围：`instruments`（默认为"csi300"）
- 特征列表：在`FilterCol`处理器中修改特征列表
- 标签定义：`label`参数定义了预测目标

### 数据集参数
在`dataset_config`中：
- `step_len`: 时间序列长度（默认为20）
- 训练/验证/测试集划分：在`segments`中调整时间范围

### 回测参数
在`port_analysis_config`中：
- 策略参数：`topk`和`n_drop`控制选股数量
- 回测初始资金：`account`
- 交易成本：`open_cost`、`close_cost`、`min_cost`

## 结果输出

训练完成后，Qlib会自动记录以下结果：
- 模型参数和权重
- 预测信号
- 信号分析结果
- 回测和投资组合表现指标

您可以通过Qlib的工作流接口访问这些结果：
```python
from qlib.workflow import R

exp = R.get_exp(experiment_name="transformer_stock_prediction")
recorder = exp.get_recorder()

# 访问记录的对象
recorded_objects = recorder.list_objects()

# 获取指标分析结果
indicator = recorder.load_object("indicator.pkl")
```

## 注意事项

1. 首次运行时，脚本会自动下载CN_DATA数据集，可能需要一些时间
2. 如需使用其他数据集，请修改`provider_uri`和`region`参数
3. 训练大型模型时，可能需要调整`batch_size`以适应您的硬件内存
4. 对于不同的预测任务，可能需要调整特征列表和标签定义