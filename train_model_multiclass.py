#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 多分类训练脚本

该脚本使用LightGBM算法训练用户行为预测模型，数据集来源于京东用户行为数据。
与二分类模型不同，该脚本训练的是多分类模型，可以预测用户的具体行为类型。
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import warnings
import argparse
from lightgbm import early_stopping, log_evaluation

warnings.filterwarnings('ignore')


# 定义行为类型映射字典
ACTION_TYPE_MAPPING = {
    0: "浏览",  # 原始值为1
    1: "购买",  # 原始值为2
    2: "收藏",  # 原始值为3
    3: "加购物车",  # 原始值为4
    4: "其他行为"  # 原始值为5
}


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
    
    # 加载全部数据
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
    # 处理时间格式
    df['action_time'] = df['action_time'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['action_time'] = pd.to_datetime(
        df['action_time'],
        format="%Y-%m-%d %H:%M:%S",
        errors='raise'
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
    
    # 定义目标变量：action_type作为多分类目标
    # 注意：LightGBM要求多分类标签从0开始，而action_type从1开始，所以需要减1
    df['target'] = df['action_type'] - 1
    
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
    print("目标变量分布:")
    for action_type, count in df['target'].value_counts().sort_index().items():
        action_name = ACTION_TYPE_MAPPING.get(action_type, f"未知行为({action_type})")
        print(f"  - {action_name}: {count} ({count/len(df):.2%})")
    
    return X, y, feature_cols


def train_lightgbm_model_multiclass(X_train, y_train, X_val, y_val, feature_names, categorical_features=None):
    """
    训练LightGBM多分类模型
    
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
    print("开始训练LightGBM多分类模型...")
    start_time = time.time()
    
    # 获取类别数量
    num_class = len(np.unique(y_train))
    print(f"类别数量: {num_class}")
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, 
                            categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, 
                          categorical_feature=categorical_features, reference=train_data)
    
    # 设置参数 - 多分类
    params = {
        'objective': 'multiclass',  # 多分类目标
        'num_class': num_class,     # 类别数量
        'metric': 'multi_logloss',  # 多分类损失函数
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
    )
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"最佳迭代次数: {model.best_iteration}")
    
    return model


def evaluate_multiclass_model(model, X, y):
    """
    评估多分类模型性能
    
    Args:
        model: 训练好的模型
        X: 特征数据
        y: 真实标签
        
    Returns:
        评估指标字典
    """
    print("评估模型性能...")
    
    # 预测概率
    y_pred_proba = model.predict(X)
    
    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)  # 不需要+1，因为我们已经将标签调整为从0开始
    
    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    # 计算每个类别的精确率、召回率和F1分数
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)
    
    # 打印每个类别的性能指标
    print("\n各类别性能指标:")
    for i, action_id in enumerate(sorted(np.unique(y))):
        action_name = ACTION_TYPE_MAPPING.get(action_id, f"未知行为({action_id})")
        idx = i  # 索引可能需要调整，取决于类别的编码方式
        if idx < len(precision):
            print(f"  - {action_name}:")
            print(f"    精确率: {precision[idx]:.4f}")
            print(f"    召回率: {recall[idx]:.4f}")
            print(f"    F1分数: {f1[idx]:.4f}")
    
    # 计算宏平均和加权平均性能指标
    macro_precision = precision_score(y, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    print("\n总体性能指标:")
    print(f"  宏平均精确率: {macro_precision:.4f}")
    print(f"  宏平均召回率: {macro_recall:.4f}")
    print(f"  宏平均F1分数: {macro_f1:.4f}")
    print(f"  加权平均精确率: {weighted_precision:.4f}")
    print(f"  加权平均召回率: {weighted_recall:.4f}")
    print(f"  加权平均F1分数: {weighted_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def plot_feature_importance(model, feature_names):
    """
    绘制特征重要性图
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
    """
    print("绘制特征重要性图...")
    
    # 获取特征重要性
    importance = model.feature_importance(importance_type='split')
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # 只显示前20个特征
    top_features = feature_importance.head(20)
    
    # 绘制条形图
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('特征重要性')
    plt.ylabel('特征')
    plt.title('LightGBM特征重要性（多分类模型）')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('plots/feature_importance_multiclass.png')
    print("特征重要性图已保存到plots/feature_importance_multiclass.png")


def save_model(model, model_path):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        model_path: 模型保存路径
    """
    print(f"保存模型到: {model_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型
    model.save_model(model_path)
    print("模型保存完成")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为多分类预测模型训练（使用分割数据集）')
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


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    
    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    print("=== 使用已分割的数据集进行多分类模型训练 ===")
    
    # 加载训练集
    print("\n=== 加载训练集 ===")
    train_df = load_data(args.train_data)
    X_train, y_train, feature_cols = preprocess_data(train_df)
    print(f"训练集大小: {X_train.shape}")
    
    # 加载验证集
    print("\n=== 加载验证集 ===")
    val_df = load_data(args.val_data)
    X_val, y_val, _ = preprocess_data(val_df)
    print(f"验证集大小: {X_val.shape}")
    
    # 加载测试集（如果需要）
    print("\n=== 加载测试集 ===")
    test_df = load_data(args.test_data)
    X_test, y_test, _ = preprocess_data(test_df)
    print(f"测试集大小: {X_test.shape}")
    
    # 检查各数据集中的类别分布
    print("\n=== 检查类别分布 ===")
    print("训练集类别分布:")
    for action_type, count in y_train.value_counts().sort_index().items():
        action_name = ACTION_TYPE_MAPPING.get(action_type, f"未知行为({action_type})")
        print(f"  - {action_name}: {count} ({count/len(y_train):.2%})")
    
    print("验证集类别分布:")
    for action_type, count in y_val.value_counts().sort_index().items():
        action_name = ACTION_TYPE_MAPPING.get(action_type, f"未知行为({action_type})")
        print(f"  - {action_name}: {count} ({count/len(y_val):.2%})")
    
    # 定义分类特征
    categorical_features = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'province', 'city', 'county']
    categorical_features = [col for col in categorical_features if col in feature_cols]
    
    # 设置模型参数
    params = {
        'objective': 'multiclass',  # 多分类目标
        'num_class': len(np.unique(y_train)),  # 类别数量
        'metric': 'multi_logloss',  # 多分类损失函数
        'boosting_type': 'gbdt',
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 训练模型
    print("\n=== 开始训练模型 ===")
    model = train_lightgbm_model_multiclass(X_train, y_train, X_val, y_val, feature_cols, categorical_features)
    
    # 评估模型在验证集上的性能
    print("\n=== 在验证集上评估模型 ===")
    val_metrics = evaluate_multiclass_model(model, X_val, y_val)
    
    # 评估模型在测试集上的性能
    print("\n=== 在测试集上评估模型 ===")
    test_metrics = evaluate_multiclass_model(model, X_test, y_test)
    
    # 绘制特征重要性
    plot_feature_importance(model, feature_cols)
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'lightgbm_user_behavior_multiclass_model.model')
    save_model(model, model_path)
    
    print("\n=== 训练完成！===")
    print("验证集性能:")
    print(f"  准确率: {val_metrics['accuracy']:.4f}")
    print(f"  宏平均F1分数: {val_metrics['macro_f1']:.4f}")
    print(f"  加权平均F1分数: {val_metrics['weighted_f1']:.4f}")
    
    print("测试集性能:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  宏平均F1分数: {test_metrics['macro_f1']:.4f}")
    print(f"  加权平均F1分数: {test_metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()