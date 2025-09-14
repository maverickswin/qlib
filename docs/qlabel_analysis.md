# qlib Label计算原理详解

## 问题背景

用户发现qlib计算出的label值变化很大，而手动计算的label值相对正常，想了解原因。

## 核心发现

**qlib的label变化大 = 原始收益率 + Z-score标准化**

## 详细分析

### 1. 原始Label计算

```python
# Alpha158的标签定义
"Ref($close, -2)/Ref($close, -1) - 1"

# 含义：未来2日相对于未来1日的收益率
# 手动计算结果示例：
# 2014-01-02: -0.034711
# 2014-01-03: -0.001142  
# 2014-01-06: -0.017714
# 2014-01-07: -0.008144
# 2014-01-08: 0.003519
```

### 2. qlib的数据处理流程

```python
# Alpha158默认的learn_processors
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},  # 关键！
]
```

### 3. CSZScoreNorm处理器详解

```python
class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""
    
    def __call__(self, df):
        # 按日期分组，对每个交易日内的所有股票进行Z-score标准化
        for g in self.fields_group:
            cols = get_group_columns(df, g)
            # 关键：按datetime分组，对每个交易日内的所有股票进行标准化
            df[cols] = df[cols].groupby("datetime", group_keys=False).apply(self.zscore_func)
        return df
```

### 4. 具体计算过程

```python
# 步骤1：计算原始标签
original_label = (Ref($close, -2) / Ref($close, -1)) - 1

# 步骤2：CSZScoreNorm处理
# 对于每个交易日t：
# 1. 获取该交易日所有股票的标签值
# 2. 计算该日所有股票标签的均值和标准差
# 3. 对每个股票的标签进行Z-score标准化：
normalized_label = (original_label - daily_mean) / daily_std
```

### 5. 数值对比示例

```python
# 假设某交易日有300只股票，标签分布如下：
daily_labels = [0.01, 0.02, 0.03, ..., -0.05, -0.03, ...]  # 300个值
daily_mean = 0.001  # 该日所有股票标签的均值
daily_std = 0.02    # 该日所有股票标签的标准差

# 原始标签为0.01的股票，标准化后变成：
normalized_label = (0.01 - 0.001) / 0.02 = 0.45

# 原始标签为-0.05的股票，标准化后变成：
normalized_label = (-0.05 - 0.001) / 0.02 = -2.55
```

### 6. 结果对比

| 类型 | 数值范围 | 含义 | 示例 |
|------|----------|------|------|
| 手动计算 | -0.1 到 +0.1 | 绝对收益率 | [-0.034711, -0.001142, ...] |
| qlib结果 | -3 到 +3+ | 相对排名 | [0.45, -0.12, 1.23, -2.55, ...] |

## 为什么Z-score会让值变大

### 1. 数值范围变化
- **原始收益率**：通常在 -0.1 到 +0.1 之间（-10% 到 +10%）
- **Z-score后**：可能达到 -3 到 +3 甚至更大

### 2. 标准差影响
- 如果某日股票表现差异很大，标准差小，Z-score值就会很大
- 如果某日股票表现趋同，标准差大，Z-score值相对较小

### 3. 截面效应
- 单只股票的表现会受到当日所有股票表现的影响
- 标准化后的值反映的是相对排名，而非绝对表现

## 实际意义对比

### 手动计算结果
- **含义**：绝对收益率，直接反映股票涨跌幅度
- **优点**：直观易懂，反映真实收益
- **缺点**：受市场整体波动影响

### qlib结果
- **含义**：相对排名，反映股票相对于当日市场的表现
- **优点**：消除市场整体波动影响，专注于个股相对表现
- **缺点**：数值不直观，需要理解标准化含义

## 结论

1. **qlib的label不是"错误"**，而是经过标准化处理的相对表现指标
2. **Z-score标准化**是导致数值变化大的根本原因
3. **两种结果都有意义**：
   - 手动计算：适合理解绝对收益
   - qlib结果：适合机器学习模型训练
4. **标准化后的label更适合ML模型**，因为它消除了市场整体波动的影响，专注于个股的相对表现

## 代码验证

```python
def simulate_cszscore_normalization(single_stock_labels, market_labels):
    """
    模拟CSZScoreNorm的处理过程
    """
    results = []
    
    for date, single_label in single_stock_labels.items():
        # 获取该日所有股票的标签
        daily_market_labels = market_labels[date]
        
        # 计算该日的均值和标准差
        daily_mean = daily_market_labels.mean()
        daily_std = daily_market_labels.std()
        
        # 标准化
        normalized_label = (single_label - daily_mean) / daily_std
        results.append(normalized_label)
    
    return results
```

## 文件位置

- **Alpha158定义**: `qlib/contrib/data/handler.py`
- **CSZScoreNorm实现**: `qlib/data/dataset/processor.py`
- **默认处理器配置**: `qlib/contrib/data/handler.py:36-44`

---
*总结：qlib通过Z-score标准化将绝对收益率转换为相对排名，这是导致label数值变化大的根本原因。*