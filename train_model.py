#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型

该脚本使用LightGBM算法训练用户行为预测模型，数据集来源于京东用户行为数据。
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import warnings
from lightgbm import early_stopping, log_evaluation

warnings.filterwarnings('ignore')


def load_data(file_path, sample_size=None):
    """
    加载数据集，由于数据集较大，可以选择采样加载
    
    Args:
        file_path: 数据文件路径
        sample_size: 采样大小，如果为None则加载全部数据
        
    Returns:
        DataFrame: 加载的数据集
    """
    print(f"开始加载数据集: {file_path}")
    start_time = time.time()
    
    if sample_size:
        # 如果指定了采样大小，则随机采样加载
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        # 否则加载全部数据
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"数据集大小: {df.shape}")
    return df


def preprocess_data(df):
    """
    数据预处理，包括特征工程、缺失值处理等
    
    Args:
        df: 原始数据集
        
    Returns:
        X: 特征数据
        y: 目标变量
    """
    print("开始数据预处理...")
    start_time = time.time()
    
    # 数据类型转换
    # df['action_time'] = pd.to_datetime(df['action_time'])
    # —— 1. 强制转成字符串，去掉末尾的 .0（如果有）
    df['action_time'] = df['action_time'].astype(str).str.replace(r'\.0$', '', regex=True)

    # —— 2. 按照“年/月/日 时:分:秒”格式解析
    df['action_time'] = pd.to_datetime(
        df['action_time'],
        format="%Y-%m-%d %H:%M:%S",
        errors='raise'   # 如果格式不对就报错，便于排查
    )
    
    # 提取时间特征
    df['hour'] = df['action_time'].dt.hour
    df['day'] = df['action_time'].dt.day
    df['month'] = df['action_time'].dt.month
    df['dayofweek'] = df['action_time'].dt.dayofweek
    
    # 处理缺失值
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(-1)
    
    # 对分类特征进行编码
    categorical_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'province', 'city', 'county']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 用户行为聚合特征
    user_stats = df.groupby('user_log_acct').agg({
        'item_sku_id': 'count',
        'action_type': 'mean',
        'shop_id': 'nunique',
        'brand_code': 'nunique',
        'item_third_cate_cd': 'nunique'
    }).reset_index()
    
    user_stats.columns = ['user_log_acct', 'user_item_count', 'user_action_mean', 
                         'user_shop_nunique', 'user_brand_nunique', 'user_cate_nunique']
    
    # 合并聚合特征
    df = pd.merge(df, user_stats, on='user_log_acct', how='left')
    
    # 商品特征聚合
    item_stats = df.groupby('item_sku_id').agg({
        'user_log_acct': 'count',
        'action_type': 'mean'
    }).reset_index()
    
    item_stats.columns = ['item_sku_id', 'item_user_count', 'item_action_mean']
    
    # 合并商品聚合特征
    df = pd.merge(df, item_stats, on='item_sku_id', how='left')
    
    # 定义目标变量：action_type为购买行为(假设action_type=2表示购买)
    # 根据数据集的实际情况调整
    df['target'] = (df['action_type'] == 2).astype(int)
    
    # 选择特征列
    feature_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'shop_score',
                   'age', 'sex', 'user_level', 'province', 'city', 'county', 'hour', 'day',
                   'month', 'dayofweek', 'user_item_count', 'user_action_mean', 'user_shop_nunique',
                   'user_brand_nunique', 'user_cate_nunique', 'item_user_count', 'item_action_mean']
    
    # 确保所有特征列都存在
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"特征数量: {len(feature_cols)}")
    print(f"正样本比例: {y.mean():.4f}")
    
    return X, y, feature_cols


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names, categorical_features=None):
    """
    训练LightGBM模型
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        feature_names: 特征名称列表
        categorical_features: 分类特征列表
        
    Returns:
        训练好的模型
    """
    print("开始训练LightGBM模型...")
    start_time = time.time()
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, 
                            categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, 
                          categorical_feature=categorical_features, reference=train_data)
    
    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
        # early_stopping_rounds=50,
        # verbose_eval=100
    )
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"最佳迭代次数: {model.best_iteration}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    print("开始评估模型性能...")
    
    # 预测概率
    y_pred_proba = model.predict(X_test)
    
    # 转换为二分类结果
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算各种评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # 打印评估结果
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return metrics


def plot_feature_importance(model, feature_names, top_n=20):
    """
    绘制特征重要性图
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
    """
    # 获取特征重要性
    importance = model.feature_importance(importance_type='split')
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][::-1], feature_importance['Importance'][::-1])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print(f"特征重要性图已保存为 'feature_importance.png'")


def save_model(model, model_path):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        model_path: 模型保存路径
    """
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")


def main():
    # 设置数据文件路径
    data_path = "data/jd_data.csv"
    
    # 加载数据（使用采样以加快处理速度）
    # 实际应用中可以根据计算资源调整采样大小或使用全量数据
    df = load_data(data_path)  # 采样100万条数据
    
    # 数据预处理
    X, y, feature_names = preprocess_data(df)
    
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 定义分类特征
    categorical_features = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 
                           'province', 'city', 'county', 'sex', 'user_level']
    categorical_features = [col for col in categorical_features if col in feature_names]
    
    # 训练模型
    model = train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names, categorical_features)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 绘制特征重要性
    plot_feature_importance(model, feature_names)
    
    # 保存模型
    save_model(model, "model/lightgbm_user_behavior_model.txt")


if __name__ == "__main__":
    main()