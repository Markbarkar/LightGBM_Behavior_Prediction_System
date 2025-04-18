#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 使用分割数据集训练

该脚本使用已经分割好的训练集、测试集和验证集训练LightGBM模型，
并绘制训练过程中的模型性能变化图表。
支持双GPU并行训练。
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import argparse
import torch
from lightgbm import early_stopping, log_evaluation
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# 设置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为预测模型训练（使用分割数据集）')
    parser.add_argument('--train_data', type=str, default='./data/processed/train.csv', help='训练集文件路径')
    parser.add_argument('--val_data', type=str, default='./data/processed/val.csv', help='验证集文件路径')
    parser.add_argument('--test_data', type=str, default='./data/processed/test.csv', help='测试集文件路径')
    parser.add_argument('--model_dir', type=str, default='./model', help='模型保存目录')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='图表保存目录')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='学习率')
    parser.add_argument('--num_leaves', type=int, default=31, help='叶子节点数')
    parser.add_argument('--max_depth', type=int, default=-1, help='最大深度，-1表示不限制')
    parser.add_argument('--num_boost_round', type=int, default=1000, help='最大迭代次数')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help='早停轮数')
    parser.add_argument('--device_type', type=str, default='gpu', help='设备类型：gpu或cpu')
    parser.add_argument('--num_gpu', type=int, default=2, help='使用的GPU数量')
    parser.add_argument('--gpu_platform_id', type=int, default=0, help='GPU平台ID')
    parser.add_argument('--gpu_device_id', type=int, default=0, help='GPU设备ID')
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
    
    Args:
        df: 原始数据集
        
    Returns:
        X: 特征数据
        y: 目标变量
        feature_cols: 特征列名
        categorical_features: 类别特征列表
    """
    print("开始数据预处理...")
    start_time = time.time()
    
    # 数据类型转换
    # 处理action_time列，确保正确的日期时间格式
    if 'action_time' in df.columns:
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
    
    # 定义类别特征
    categorical_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 
                       'province', 'city', 'county', 'sex', 'user_level', 'hour', 
                       'day', 'month', 'dayofweek']
    
    # 对分类特征进行编码
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
    
    # 确保类别特征在特征列中
    categorical_features = [col for col in categorical_cols if col in feature_cols]
    
    X = df[feature_cols]
    
    # 获取目标变量（如果存在）
    y = None
    if 'target' in df.columns:
        y = df['target']
        # 检查目标变量的分布
        print(f"目标变量分布:\n{y.value_counts(normalize=True)}")
    
    # 检查特征的数据类型
    print("\n特征数据类型:")
    print(X.dtypes)
    
    print(f"\n数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"特征数量: {len(feature_cols)}")
    print(f"类别特征数量: {len(categorical_features)}")
    if y is not None:
        print(f"正样本比例: {y.mean():.4f}")
    
    return X, y, feature_cols, categorical_features


class LGBMMetricsCallback(object):
    """
    自定义回调函数，用于记录训练过程中的评估指标
    """
    def __init__(self, valid_sets, valid_names, eval_metrics=['auc', 'binary_logloss']):
        self.valid_sets = valid_sets
        self.valid_names = valid_names
        self.eval_metrics = eval_metrics
        self.metrics_history = {}
        
        for name in valid_names:
            self.metrics_history[name] = {}
            for metric in eval_metrics:
                self.metrics_history[name][metric] = []
    
    def __call__(self, env):
        for i, (name, valid_set) in enumerate(zip(self.valid_names, self.valid_sets)):
            for metric in self.eval_metrics:
                score = env.evaluation_result_list[i][2]
                self.metrics_history[name][metric].append(score)
        return False


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names, categorical_features=None, params=None, args=None):
    """
    训练LightGBM模型
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        feature_names: 特征名称列表
        categorical_features: 类别特征列表
        params: 模型参数
        args: 命令行参数
        
    Returns:
        model: 训练好的模型
        metrics_history: 训练过程中的指标历史
    """
    print("开始训练模型...")
    start_time = time.time()
    
    # 检查数据
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"类别特征数量: {len(categorical_features) if categorical_features else 0}")
    
    # 设置默认参数
    if params is None:
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'min_sum_hessian_in_leaf': 1e-3,
            'max_bin': 255,
            'num_threads': 96,  # 使用所有可用的CPU线程
            'device_type': 'cpu',
            'tree_learner': 'data_parallel',  # 使用数据并行
            'histogram_pool_size': 2048,  # 增加直方图池大小
            'max_depth': 6,  # 限制树深度以提高训练速度
            'min_gain_to_split': 0.1,  # 增加分裂增益阈值
            'lambda_l1': 0.1,  # L1正则化
            'lambda_l2': 0.1,  # L2正则化
            'feature_fraction_seed': 42,  # 固定特征采样种子
            'bagging_seed': 42,  # 固定bagging种子
            'data_random_seed': 42  # 固定数据随机种子
        }
    
    # 如果提供了args，更新相关参数
    if args is not None:
        params.update({
            'device_type': args.device_type,
            'num_threads': 96 if args.device_type == 'cpu' else 0,
            'num_leaves': args.num_leaves,
            'learning_rate': args.learning_rate,
            'num_boost_round': args.num_boost_round
        })
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                           categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names,
                         categorical_feature=categorical_features)
    
    # 创建指标记录回调
    metrics_callback = LGBMMetricsCallback(
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        eval_metrics=['auc', 'binary_logloss']
    )
    
    # 定义回调函数
    callbacks = [
        early_stopping(args.early_stopping_rounds),
        log_evaluation(period=10),
        metrics_callback
    ]
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    return model, metrics_callback.metrics_history


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


def plot_learning_curves(metrics_history, plot_dir):
    """
    绘制学习曲线
    
    Args:
        metrics_history: 训练过程中的指标历史
        plot_dir: 图表保存目录
    """
    print("绘制学习曲线...")
    
    # 创建图表目录
    os.makedirs(plot_dir, exist_ok=True)
    
    # 绘制训练和验证集的AUC曲线
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_history['train']['auc'], label='Train AUC')
    plt.plot(metrics_history['valid']['auc'], label='Validation AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'auc_curve.png'))
    plt.close()
    
    # 绘制训练和验证集的损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_history['train']['binary_logloss'], label='Train Loss')
    plt.plot(metrics_history['valid']['binary_logloss'], label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_curve.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, plot_dir):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        plot_dir: 图表保存目录
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()


def plot_feature_importance(model, feature_names, plot_dir, top_n=20):
    """
    绘制特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        plot_dir: 图表保存目录
        top_n: 显示前N个重要特征
    """
    print("绘制特征重要性...")
    
    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
    plt.title('Feature Importance (Top {})'.format(top_n))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
    plt.close()
    
    # 保存特征重要性到CSV文件
    feature_importance.to_csv(os.path.join(plot_dir, 'feature_importance.csv'), index=False)


def plot_metrics_comparison(metrics, plot_dir):
    """
    绘制模型评估指标对比
    
    Args:
        metrics: 评估指标字典
        plot_dir: 图表保存目录
    """
    print("绘制评估指标对比...")
    
    # 准备数据
    metrics_data = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', data=metrics_data)
    plt.title('Model Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metrics_comparison.png'))
    plt.close()


def save_model(model, model_path):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        model_path: 模型保存路径
    """
    print(f"保存模型到: {model_path}")
    model.save_model(model_path)
    
    # 保存模型参数
    params_path = model_path.replace('.txt', '_params.json')
    with open(params_path, 'w') as f:
        json.dump(model.params, f, indent=4)


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置CPU训练
    args.device_type = 'cpu'
    print(f"使用CPU训练，可用线程数: 96")
    
    # 加载数据
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    # 数据预处理
    X_train, y_train, feature_cols, categorical_features = preprocess_data(train_df)
    X_val, y_val, _, _ = preprocess_data(val_df)
    X_test, y_test, _, _ = preprocess_data(test_df)
    
    # 训练模型
    model, metrics_history = train_lightgbm_model(X_train, y_train, X_val, y_val, feature_cols, 
                                                categorical_features=categorical_features, args=args)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'model.txt')
    save_model(model, model_path)
    
    # 绘制各种图表
    plot_learning_curves(metrics_history, args.plot_dir)
    plot_feature_importance(model, feature_cols, args.plot_dir)
    plot_metrics_comparison(metrics, args.plot_dir)
    
    # 绘制混淆矩阵
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred_binary, args.plot_dir)
    
    print("训练完成！")
    print(f"模型已保存到: {model_path}")
    print(f"图表已保存到: {args.plot_dir}")


if __name__ == '__main__':
    main()