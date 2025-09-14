## 概述

Alpha158是qlib中一个158维特征的数据集，主要用于量化投资中的机器学习任务。本文档详细记录了Alpha158从配置到数据准备完成的完整流程。

## 1. 配置阶段

### 1.1 用户配置文件
```yaml
# workflow_config_lightgbm_Alpha158.yaml
dataset:
  class: DatasetH
  kwargs:
    handler:
      class: Alpha158
      kwargs:
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: csi300
        infer_processors:
          - class: RobustZScoreNorm
            kwargs:
              fields_group: feature
              clip_outlier: true
          - class: Fillna
            kwargs:
              fields_group: feature
        learn_processors:
          - class: DropnaLabel
          - class: CSRankNorm
            kwargs:
              fields_group: label
          - class: CSZScoreNorm
            kwargs:
              fields_group: feature
```

### 1.2 Alpha158配置
```python
# qlib/contrib/data/handler.py
def get_feature_config(self):
    conf = {
        \"kbar\": {},                    # K线特征
        \"price\": {                     # 价格特征
            \"windows\": [0],
            \"feature\": [\"OPEN\", \"HIGH\", \"LOW\", \"VWAP\"],
        },
        \"rolling\": {},                 # 滚动特征
    }
    return conf
```

## 2. 初始化阶段

### 2.1 Alpha158实例化
```python
# 1. 创建Alpha158实例
alpha158 = Alpha158(
    start_time=\"2008-01-01\",
    end_time=\"2020-08-01\",
    instruments=\"csi300\"
)

# 2. 调用get_feature_config()生成特征配置
feature_config = alpha158.get_feature_config()

# 3. 调用parse_config_to_fields()解析配置
fields = alpha158.parse_config_to_fields(feature_config)
```

## 3. 数据加载阶段

### 3.1 调用链
```python
# 1. 模型调用
model.fit(dataset) 
    ↓
# 2. 模型内部调用
ds_l = self._prepare_data(dataset, reweighter)
    ↓
# 3. _prepare_data 调用
df = dataset.prepare(key, col_set=[\"feature\", \"label\"], data_key=DataHandlerLP.DK_L)
    ↓
# 4. dataset.prepare 调用
return self.handler.fetch(col_set=col_set, data_key=data_key, segments=segments)
    ↓
# 5. Alpha158.fetch 调用 (继承自 DataHandlerLP)
return self._fetch_data(data_storage=self._get_df_by_key(data_key), ...)
    ↓
# 6. _get_df_by_key 返回处理后的数据
df = getattr(self, self.ATTR_MAP[data_key])  # 返回 self._learn 或 self._infer
```

## 4. 特征计算详解

### 4.1 K线特征 (9个)
- **KMID**: `($close-$open)/$open` - 收盘价相对开盘价的变化
- **KLEN**: `($high-$low)/$open` - 高低价差相对开盘价
- **KMID2**: `($close-$open)/($high-$low+1e-12)` - 实体相对影线比例
- **KUP**: `($high-Greater($open, $close))/$open` - 上影线相对开盘价
- **KUP2**: `($high-Greater($open, $close))/($high-$low+1e-12)` - 上影线相对总长度
- **KLOW**: `(Less($open, $close)-$low)/$open` - 下影线相对开盘价
- **KLOW2**: `(Less($open, $close)-$low)/($high-$low+1e-12)` - 下影线相对总长度
- **KSFT**: `(2*$close-$high-$low)/$open` - 收盘价相对高低价中点
- **KSFT2**: `(2*$close-$high-$low)/($high-$low+1e-12)` - 收盘价相对高低价中点的比例

### 4.2 价格特征 (4个)
- **OPEN0**: 当前交易日开盘价
- **HIGH0**: 当前交易日最高价
- **LOW0**: 当前交易日最低价
- **VWAP0**: 当前交易日成交量加权平均价

### 4.3 滚动特征 (大量)
- **移动平均**: `Mean($close, 5)`, `Mean($close, 10)`, `Mean($close, 20)`
- **移动标准差**: `Std($close, 5)`, `Std($close, 10)`, `Std($close, 20)`
- **移动相关性**: `Corr($close, $volume, 5)`, `Corr($close, $volume, 10)`
- **移动排名**: `Rank($close, 5)`, `Rank($close, 10)`
- **移动极值**: `Max($high, 5)`, `Min($low, 5)`

## 5. 标签计算

### 5.1 标签配置
```python
def get_label_config(self):
    return ([\"Ref($close, -2)/Ref($close, -1) - 1\"], [\"LABEL0\"])
```

### 5.2 标签含义
- **LABEL0**: `Ref($close, -2)/Ref($close, -1) - 1`
  - 表示未来2日相对于未来1日的收益率
  - 用于预测股票的未来表现

## 6. 数据预处理

### 6.1 学习阶段处理器
```python
learn_processors = [
    DropnaLabel(),           # 删除标签为NaN的样本
    CSRankNorm(fields_group=\"label\"),  # 截面排名标准化
    CSZScoreNorm(fields_group=\"feature\")  # 截面Z-score标准化
]
```

### 6.2 推理阶段处理器
```python
infer_processors = [
    RobustZScoreNorm(        # 鲁棒Z-score标准化
        fields_group=\"feature\",
        clip_outlier=True
    ),
    Fillna(fields_group=\"feature\")  # 填充缺失值
]
```

## 7. 最终输出

### 7.1 数据格式
```python
# 最终输出的DataFrame格式
df = pd.DataFrame({
    \"feature\": [158维特征数据],
    \"label\": [标签数据],
    \"datetime\": [时间索引],
    \"instrument\": [股票代码]
})
```

### 7.2 特征维度
- **总特征数**: 158维
- **K线特征**: 9维
- **价格特征**: 4维
- **滚动特征**: 145维 (各种统计指标的组合)

## 8. 关键文件位置

- **Alpha158定义**: `qlib/contrib/data/handler.py`
- **特征解析**: `qlib/contrib/data/handler.py:parse_config_to_fields()`
- **数据加载**: `qlib/data/dataset/loader.py`
- **数据处理**: `qlib/data/dataset/handler.py`
- **配置文件**: `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`

## 9. 数据导出方法

### 9.1 在GBDT模型中导出数据
```python
def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
    ds_l = []
    assert \"train\" in dataset.segments
    for key in [\"train\", \"valid\"]:
        if key in dataset.segments:
            df = dataset.prepare(key, col_set=[\"feature\", \"label\"], data_key=DataHandlerLP.DK_L)
            if df.empty:
                raise ValueError(\"Empty data from dataset, please check your dataset config.\")
            
            # 导出原始数据
            df.to_parquet(f\"./{key}_raw_data.parquet\")
            print(f\"已导出 {key} 原始数据到 ./{key}_raw_data.parquet\")
            
            x, y = df[\"feature\"], df[\"label\"]
            # ... 其余代码保持不变
```

### 9.2 导出文件说明
- `./train_raw_data.parquet` - 训练集原始数据
- `./valid_raw_data.parquet` - 验证集原始数据
- 包含完整的特征、标签、时间索引和股票代码信息

## 10. 总结

Alpha158的数据准备过程是一个复杂的流水线，包括：
1. **配置解析**: 将简单配置转换为复杂的特征表达式
2. **特征计算**: 基于原始数据计算158维特征
3. **标签生成**: 计算未来收益率标签
4. **数据预处理**: 标准化、去噪、填充缺失值
5. **数据输出**: 生成可用于机器学习的数据格式

整个过程体现了qlib框架的灵活性和可扩展性，通过简单的配置就能生成复杂的量化特征。

## 11. 关键理解点

### 11.1 配置的\"简单\"是假象
虽然传入的配置看起来很简单，但`parse_config_to_fields`函数会根据配置自动生成大量的特征表达式。

### 11.2 数据转换占主要effort
在qlib的整个流程中，数据转换和特征工程占了绝大部分的工作量，而实际的模型训练部分相对简单。

### 11.3 模块化设计
数据准备和模型训练完全分离，同一套特征可以用于不同模型，同一套数据可以用于不同策略。

---
*本文档总结了Alpha158从配置到数据准备的完整流程，帮助理解qlib的数据处理机制。*