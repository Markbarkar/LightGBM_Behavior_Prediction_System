# 基于LightGBM的用户行为预测模型

## 项目简介

本项目使用LightGBM算法对京东用户行为数据进行分析和预测，旨在预测用户的购买行为。通过对用户历史行为数据的学习，模型能够预测用户是否会对特定商品进行购买操作。

## 数据集说明

数据集位于`data/jd_data.csv`，包含京东用户的行为数据，主要字段包括：

- `user_log_acct`: 用户ID
- `item_sku_id`: 商品ID
- `action_time`: 行为时间
- `action_type`: 行为类型（如浏览、购买等）
- `brand_code`: 品牌编码
- `shop_id`: 店铺ID
- `item_third_cate_cd`: 商品三级类目编码
- `vender_id`: 供应商ID
- `shop_score`: 店铺评分
- `age`: 用户年龄
- `sex`: 用户性别
- `user_level`: 用户等级
- `province`: 省份
- `city`: 城市
- `county`: 区县

## 功能特点

- 数据预处理：处理缺失值、特征编码、时间特征提取
- 特征工程：构建用户行为聚合特征、商品特征
- 模型训练：使用LightGBM算法训练二分类模型
- 模型评估：计算准确率、精确率、召回率、F1分数和AUC等指标
- 特征重要性分析：可视化展示最重要的特征

## 使用方法

### 环境准备

```bash
pip install -r requirements.txt
```

### 模型训练

```bash
python train_model.py
```

默认情况下，脚本会从数据集中采样100万条记录进行训练。如需使用全量数据，请修改`train_model.py`中的`sample_size`参数。

### 模型预测

```bash
python predict.py --input <input_file> --output <output_file>
```

## 依赖项

- Python 3.6+
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib

## 模型性能

在测试集上，模型的主要性能指标如下：
- 准确率(Accuracy): 根据实际训练结果而定
- AUC: 根据实际训练结果而定
- F1分数: 根据实际训练结果而定

## 注意事项

- 数据集较大（约37GB），请确保有足够的内存和存储空间
- 默认使用采样数据进行训练，全量训练可能需要较长时间
- 模型训练完成后会保存为`lightgbm_user_behavior_model.txt`