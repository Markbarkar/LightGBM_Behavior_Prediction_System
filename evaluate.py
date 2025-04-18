#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 模型评估脚本

该脚本用于加载训练好的LightGBM模型，并对测试集或验证集进行评估，计算各种性能指标。
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import time
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为预测模型评估')
    parser.add_argument('--input', type=str, required=False, help='输入数据文件路径', default='./data/processed/test.csv')
    parser.add_argument('--model', type=str, default='model/lightgbm_user_behavior_model.txt', help='模型文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='预测概率阈值')
    return parser.parse_args()


def load_data(file_path):
    """
    加载数据集
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        DataFrame: 加载的数据集
    """
    print(f"开始加载数据集: {file_path}")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"数据集大小: {df.shape}")
    return df


def preprocess_data(df):
    """
    数据预处理，包括特征工程、缺失值处理等
    与训练脚本中的预处理保持一致
    
    Args:
        df: 原始数据集
        
    Returns:
        X: 特征数据
        y: 目标变量
        feature_cols: 特征列名
    """
    print("开始数据预处理...")
    start_time = time.time()
    
    # 数据类型转换
    df['action_time'] = pd.to_datetime(df['action_time'])
    
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
    
    # 选择特征列
    feature_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'shop_score',
                   'age', 'sex', 'user_level', 'province', 'city', 'county', 'hour', 'day',
                   'month', 'dayofweek', 'user_item_count', 'user_action_mean', 'user_shop_nunique',
                   'user_brand_nunique', 'user_cate_nunique', 'item_user_count', 'item_action_mean']
    
    # 确保所有特征列都存在
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    
    # 获取目标变量（如果存在）
    y = None
    if 'target' in df.columns:
        y = df['target']
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"特征数量: {len(feature_cols)}")
    
    return X, y, feature_cols


def load_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    print(f"加载模型: {model_path}")
    model = lgb.Booster(model_file=model_path)
    return model


def evaluate_model(model, X, y, threshold=0.5):
    """
    评估模型性能
    
    Args:
        model: 加载的模型
        X: 特征数据
        y: 真实标签
        threshold: 预测概率阈值
        
    Returns:
        评估指标字典
    """
    print("开始评估模型...")
    start_time = time.time()
    
    # 预测概率
    y_pred_proba = model.predict(X)
    
    # 转换为二分类结果
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_pred_proba)
    }
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp
    
    print(f"模型评估完成，耗时: {time.time() - start_time:.2f}秒")
    
    return metrics


def print_evaluation_results(metrics):
    """
    打印评估结果
    
    Args:
        metrics: 评估指标字典
    """
    print("\n模型评估结果:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    print("\n混淆矩阵:")
    print(f"真阳性 (True Positive): {metrics['true_positive']}")
    print(f"假阳性 (False Positive): {metrics['false_positive']}")
    print(f"假阴性 (False Negative): {metrics['false_negative']}")
    print(f"真阴性 (True Negative): {metrics['true_negative']}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载数据
    df = load_data(args.input)
    
    # 数据预处理
    X, y, feature_cols = preprocess_data(df)
    
    # 检查是否有目标变量
    if y is None:
        print("错误: 输入数据中没有目标变量 'target'，无法评估模型性能。")
        return
    
    # 加载模型
    model = load_model(args.model)
    
    # 评估模型
    metrics = evaluate_model(model, X, y, args.threshold)
    
    # 打印评估结果
    print_evaluation_results(metrics)
    
    print("模型评估完成！")


if __name__ == "__main__":
    main()