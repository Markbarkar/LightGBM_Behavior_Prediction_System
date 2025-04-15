#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 使用分割数据集训练

该脚本使用已经分割好的训练集、测试集和验证集训练LightGBM模型，
并绘制训练过程中的模型性能变化图表。
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import argparse
from lightgbm import early_stopping, log_evaluation

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
    if y is not None:
        print(f"正样本比例: {y.mean():.4f}")
    
    return X, y, feature_cols


class LGBMMetricsCallback(object):
    """
    自定义回调函数，用于记录训练过程中的评估指标
    """
    def __init__(self, valid_sets, valid_names, eval_metrics=['auc']):
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


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names, categorical_features=None, params=None):
    """
    训练LightGBM模型
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        feature_names: 特征名称列表
        categorical_features: 分类特征列表
        params: 模型参数
        
    Returns:
        训练好的模型和训练历史记录
    """
    print("开始训练LightGBM模型...")
    start_time = time.time()
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, 
                            categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, 
                          categorical_feature=categorical_features, reference=train_data)
    
    # 设置默认参数
    if params is None:
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
    
    # 创建自定义回调函数，用于记录训练过程中的评估指标
    metrics_callback = LGBMMetricsCallback(
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        eval_metrics=['auc']
    )
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params.get('num_boost_round', 1000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
            log_evaluation(period=100),
            metrics_callback
        ]
    )
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"最佳迭代次数: {model.best_iteration}")
    
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
        metrics_history: 训练过程中记录的评估指标历史
        plot_dir: 图表保存目录
    """
    print("绘制学习曲线...")
    
    # 确保输出目录存在
    os.makedirs(plot_dir, exist_ok=True)
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 绘制AUC学习曲线
    plt.plot(metrics_history['train']['auc'], label='训练集AUC')
    plt.plot(metrics_history['valid']['auc'], label='验证集AUC')
    
    plt.xlabel('迭代次数')
    plt.ylabel('AUC')
    plt.title('LightGBM模型训练过程中的AUC变化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    learning_curve_path = os.path.join(plot_dir, 'learning_curve_auc.png')
    plt.savefig(learning_curve_path)
    print(f"学习曲线已保存到: {learning_curve_path}")
    plt.close()


def plot_feature_importance(model, feature_names, plot_dir, top_n=20):
    """
    绘制特征重要性图
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        plot_dir: 图表保存目录
        top_n: 显示前N个重要特征
    """
    print("绘制特征重要性图...")
    
    # 确保输出目录存在
    os.makedirs(plot_dir, exist_ok=True)
    
    # 获取特征重要性
    importance = model.feature_importance(importance_type='split')
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 绘制条形图
    ax = sns.barplot(x='Importance', y='Feature', data=feature_importance[::-1], palette='viridis')
    
    # 添加数值标签
    for i, v in enumerate(feature_importance['Importance'][::-1]):
        ax.text(v + 5, i, f"{v:.0f}", va='center')
    
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title(f'Top {top_n} 特征重要性')
    plt.tight_layout()
    
    # 保存图表
    feature_importance_path = os.path.join(plot_dir, 'feature_importance.png')
    plt.savefig(feature_importance_path)
    print(f"特征重要性图已保存到: {feature_importance_path}")
    plt.close()


def plot_metrics_comparison(metrics, plot_dir):
    """
    绘制评估指标对比图
    
    Args:
        metrics: 评估指标字典
        plot_dir: 图表保存目录
    """
    print("绘制评估指标对比图...")
    
    # 确保输出目录存在
    os.makedirs(plot_dir, exist_ok=True)
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 绘制条形图
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    ax = sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
    
    # 添加数值标签
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.ylim(0, 1.1)
    plt.xlabel('评估指标')
    plt.ylabel('值')
    plt.title('模型评估指标对比')
    plt.tight_layout()
    
    # 保存图表
    metrics_comparison_path = os.path.join(plot_dir, 'metrics_comparison.png')
    plt.savefig(metrics_comparison_path)
    print(f"评估指标对比图已保存到: {metrics_comparison_path}")
    plt.close()


def save_model(model, model_path):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        model_path: 模型保存路径
    """
    # 确保模型目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # 加载数据
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    # 数据预处理
    X_train, y_train, feature_names = preprocess_data(train_df)
    X_val, y_val, _ = preprocess_data(val_df)
    X_test, y_test, _ = preprocess_data(test_df)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 定义分类特征
    categorical_features = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 
                           'province', 'city', 'county', 'sex', 'user_level']
    categorical_features = [col for col in categorical_features if col in feature_names]
    
    # 设置模型参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_boost_round': args.num_boost_round,
        'early_stopping_rounds': args.early_stopping_rounds
    }
    
    # 训练模型
    model, metrics_history = train_lightgbm_model(
        X_train, y_train, X_val, y_val, feature_names, categorical_features, params
    )
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 绘制学习曲线
    plot_learning_curves(metrics_history, args.plot_dir)
    
    # 绘制特征重要性
    plot_feature_importance(model, feature_names, args.plot_dir)
    
    # 绘制评估指标对比图
    plot_metrics_comparison(metrics, args.plot_dir)
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'lightgbm_user_behavior_model_split.txt')
    save_model(model, model_path)


if __name__ == "__main__":
    main()