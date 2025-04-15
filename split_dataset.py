#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 数据集分割脚本

该脚本用于将原始数据集分割为训练集、测试集和验证集，并保存到指定目录。
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import time
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为预测数据集分割')
    parser.add_argument('--input', type=str, default='./data/jd_data.csv', help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./data/processed/', help='输出目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--sample_size', type=int, default=None, help='采样大小，如果为None则加载全部数据')
    return parser.parse_args()


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
    与训练脚本中的预处理保持一致
    
    Args:
        df: 原始数据集
        
    Returns:
        处理后的数据集
    """
    print("开始数据预处理...")
    start_time = time.time()
    
    # 数据类型转换
    # 1. 强制转成字符串，去掉末尾的 .0（如果有）
    df['action_time'] = df['action_time'].astype(str).str.replace(r'\.0$', '', regex=True)

    # 2. 按照"年/月/日 时:分:秒"格式解析
    df['action_time'] = pd.to_datetime(
        df['action_time'],
        format="%Y-%m-%d %H:%M:%S",
        errors='coerce'   # 使用coerce而不是raise，以便处理可能的格式错误
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
    
    # 定义目标变量：action_type为购买行为(假设action_type=2表示购买)
    df['target'] = (df['action_type'] == 2).astype(int)
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"正样本比例: {df['target'].mean():.4f}")
    
    return df


def split_dataset(df, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, random_state=42):
    """
    将数据集分割为训练集、测试集和验证集
    
    Args:
        df: 数据集
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        random_state: 随机种子
        
    Returns:
        train_df: 训练集
        test_df: 测试集
        val_df: 验证集
    """
    print("开始数据集分割...")
    
    # 检查比例之和是否为1
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 首先分割出训练集
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state, 
        stratify=df['target']
    )
    
    # 然后从剩余数据中分割出测试集和验证集
    # 计算测试集在剩余数据中的比例
    test_ratio_adjusted = test_ratio / (test_ratio + val_ratio)
    
    test_df, val_df = train_test_split(
        temp_df, 
        train_size=test_ratio_adjusted, 
        random_state=random_state, 
        stratify=temp_df['target']
    )
    
    print(f"数据集分割完成")
    print(f"训练集大小: {train_df.shape}, 正样本比例: {train_df['target'].mean():.4f}")
    print(f"测试集大小: {test_df.shape}, 正样本比例: {test_df['target'].mean():.4f}")
    print(f"验证集大小: {val_df.shape}, 正样本比例: {val_df['target'].mean():.4f}")
    
    return train_df, test_df, val_df


def save_datasets(train_df, test_df, val_df, output_dir):
    """
    保存分割后的数据集
    
    Args:
        train_df: 训练集
        test_df: 测试集
        val_df: 验证集
        output_dir: 输出目录路径
    """
    print(f"开始保存数据集到: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据集
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"训练集已保存到: {train_path}")
    print(f"测试集已保存到: {test_path}")
    print(f"验证集已保存到: {val_path}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载数据
    df = load_data(args.input, args.sample_size)
    
    # 数据预处理
    df = preprocess_data(df)
    
    # 数据集分割
    train_df, test_df, val_df = split_dataset(
        df, 
        args.train_ratio, 
        args.test_ratio, 
        args.val_ratio, 
        args.random_state
    )
    
    # 保存数据集
    save_datasets(train_df, test_df, val_df, args.output_dir)
    
    print("数据集分割和保存完成！")


if __name__ == "__main__":
    main()