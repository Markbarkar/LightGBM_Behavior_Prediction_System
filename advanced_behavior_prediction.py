#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级用户行为预测模型 - 多模型集成训练脚本

该脚本使用多种高级算法（XGBoost、CatBoost、LightGBM）进行集成学习，
并包含丰富的特征工程和可视化分析。
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
import argparse
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# 定义行为类型映射字典
ACTION_TYPE_MAPPING = {
    0: "浏览",
    1: "购买",
    2: "收藏",
    3: "加购物车",
    4: "其他行为"
}

# 设置随机种子
np.random.seed(42)

def load_data(file_path):
    """加载数据集"""
    print(f"开始加载数据集: {file_path}")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"数据集大小: {df.shape}")
    
    return df

def advanced_feature_engineering(df):
    """高级特征工程"""
    print("开始高级特征工程...")
    start_time = time.time()
    
    # 时间特征
    df['action_time'] = pd.to_datetime(df['action_time'])
    df['hour'] = df['action_time'].dt.hour
    df['day'] = df['action_time'].dt.day
    df['month'] = df['action_time'].dt.month
    df['dayofweek'] = df['action_time'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_category'] = pd.cut(df['hour'], 
                                bins=[0, 6, 12, 18, 24],
                                labels=['凌晨', '上午', '下午', '晚上'])
    
    # 用户行为统计特征
    try:
        user_stats = df.groupby('user_log_acct').agg({
            'item_sku_id': ['count', 'nunique'],
            'action_type': ['mean', 'std', 'count'],
            'shop_id': 'nunique',
            'brand_code': 'nunique',
            'item_third_cate_cd': 'nunique',
            'hour': ['mean', 'std']
        }).reset_index()
        
        user_stats.columns = ['user_log_acct'] + [f'user_{col[0]}_{col[1]}' for col in user_stats.columns[1:]]
        df = df.merge(user_stats, on='user_log_acct', how='left')
    except Exception as e:
        print(f"警告: 计算用户行为统计特征时出错: {str(e)}")
        print("将跳过用户行为统计特征的计算...")
    
    # 商品行为统计特征
    try:
        item_stats = df.groupby('item_sku_id').agg({
            'user_log_acct': ['count', 'nunique'],
            'action_type': ['mean', 'std', 'count'],
            'shop_id': 'nunique',
            'brand_code': 'nunique',
            'hour': ['mean', 'std']
        }).reset_index()
        
        item_stats.columns = ['item_sku_id'] + [f'item_{col[0]}_{col[1]}' for col in item_stats.columns[1:]]
        df = df.merge(item_stats, on='item_sku_id', how='left')
    except Exception as e:
        print(f"警告: 计算商品行为统计特征时出错: {str(e)}")
        print("将跳过商品行为统计特征的计算...")
    
    # 店铺行为统计特征
    try:
        shop_stats = df.groupby('shop_id').agg({
            'user_log_acct': ['count', 'nunique'],
            'action_type': ['mean', 'std', 'count'],
            'item_sku_id': 'nunique',
            'brand_code': 'nunique',
            'hour': ['mean', 'std']
        }).reset_index()
        
        shop_stats.columns = ['shop_id'] + [f'shop_{col[0]}_{col[1]}' for col in shop_stats.columns[1:]]
        df = df.merge(shop_stats, on='shop_id', how='left')
    except Exception as e:
        print(f"警告: 计算店铺行为统计特征时出错: {str(e)}")
        print("将跳过店铺行为统计特征的计算...")
    
    # 交互特征 - 使用安全的列名访问
    try:
        # 用户-商品交互特征
        if 'user_item_sku_id_count' in df.columns and 'user_item_sku_id_nunique' in df.columns:
            df['user_item_ratio'] = df['user_item_sku_id_count'] / df['user_item_sku_id_nunique']
        else:
            # 使用替代方法计算用户-商品比率
            df['user_item_ratio'] = df.groupby('user_log_acct')['item_sku_id'].transform('count') / \
                                  df.groupby('user_log_acct')['item_sku_id'].transform('nunique')
        
        # 用户-店铺交互特征
        if 'user_shop_id_count' in df.columns and 'user_shop_id_nunique' in df.columns:
            df['user_shop_ratio'] = df['user_shop_id_count'] / df['user_shop_id_nunique']
        else:
            # 使用替代方法计算用户-店铺比率
            df['user_shop_ratio'] = df.groupby('user_log_acct')['shop_id'].transform('count') / \
                                  df.groupby('user_log_acct')['shop_id'].transform('nunique')
        
        # 商品-用户交互特征
        if 'item_user_log_acct_count' in df.columns and 'item_user_log_acct_nunique' in df.columns:
            df['item_user_ratio'] = df['item_user_log_acct_count'] / df['item_user_log_acct_nunique']
        else:
            # 使用替代方法计算商品-用户比率
            df['item_user_ratio'] = df.groupby('item_sku_id')['user_log_acct'].transform('count') / \
                                  df.groupby('item_sku_id')['user_log_acct'].transform('nunique')
    except Exception as e:
        print(f"警告: 计算交互特征时出错: {str(e)}")
        print("将使用替代方法计算交互特征...")
        
        # 使用替代方法计算所有交互特征
        df['user_item_ratio'] = df.groupby('user_log_acct')['item_sku_id'].transform('count') / \
                              df.groupby('user_log_acct')['item_sku_id'].transform('nunique')
        df['user_shop_ratio'] = df.groupby('user_log_acct')['shop_id'].transform('count') / \
                              df.groupby('user_log_acct')['shop_id'].transform('nunique')
        df['item_user_ratio'] = df.groupby('item_sku_id')['user_log_acct'].transform('count') / \
                              df.groupby('item_sku_id')['user_log_acct'].transform('nunique')
    
    # 时间窗口特征
    try:
        df['hour_diff'] = df.groupby('user_log_acct')['hour'].diff()
        df['action_frequency'] = df.groupby('user_log_acct')['action_time'].diff().dt.total_seconds()
    except Exception as e:
        print(f"警告: 计算时间窗口特征时出错: {str(e)}")
        print("将跳过时间窗口特征的计算...")
    
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"特征工程完成，耗时: {time.time() - start_time:.2f}秒")
    return df

def plot_class_distribution(y, title="类别分布"):
    """绘制类别分布图"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("行为类型")
    plt.ylabel("样本数量")
    plt.xticks(range(len(ACTION_TYPE_MAPPING)), [ACTION_TYPE_MAPPING[i] for i in range(len(ACTION_TYPE_MAPPING))])
    plt.savefig(f'plots/{title}.png')
    plt.close()

def plot_feature_importance(models, feature_names, model_names):
    """绘制特征重要性图"""
    plt.figure(figsize=(15, 10))
    for i, (model, name) in enumerate(zip(models, model_names)):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            importance = model.feature_importance()
        else:
            continue
            
        plt.subplot(2, 2, i+1)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{name} - 特征重要性')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title="混淆矩阵"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ACTION_TYPE_MAPPING[i] for i in range(len(ACTION_TYPE_MAPPING))],
                yticklabels=[ACTION_TYPE_MAPPING[i] for i in range(len(ACTION_TYPE_MAPPING))])
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'plots/{title}.png')
    plt.close()

def plot_learning_curves(models, X_train, y_train, X_val, y_val, model_names):
    """绘制学习曲线"""
    plt.figure(figsize=(15, 10))
    for i, (model, name) in enumerate(zip(models, model_names)):
        if hasattr(model, 'evals_result'):
            train_metric = list(model.evals_result()['validation_0'].values())[0]
            val_metric = list(model.evals_result()['validation_1'].values())[0]
            
            plt.subplot(2, 2, i+1)
            plt.plot(train_metric, label='训练集')
            plt.plot(val_metric, label='验证集')
            plt.title(f'{name} - 学习曲线')
            plt.xlabel('迭代次数')
            plt.ylabel('损失值')
            plt.legend()
    plt.tight_layout()
    plt.savefig('plots/learning_curves.png')
    plt.close()

def train_models(X_train, y_train, X_val, y_val, feature_names, categorical_features):
    """训练多个模型"""
    print("开始训练多个模型...")
    start_time = time.time()
    
    # XGBoost参数
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
    
    # CatBoost参数
    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.01,
        'depth': 8,
        'l2_leaf_reg': 0.1,
        'random_seed': 42,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'cat_features': categorical_features
    }
    
    # LightGBM参数
    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    }
    
    # 训练XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 verbose=False)
    
    # 训练CatBoost
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False)
    
    # 训练LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 early_stopping_rounds=50,
                 verbose=False)
    
    # 创建投票分类器
    voting_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('cb', cb_model),
            ('lgb', lgb_model)
        ],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
    return [xgb_model, cb_model, lgb_model, voting_model]

def evaluate_models(models, X, y, model_names):
    """评估多个模型"""
    print("评估模型性能...")
    results = {}
    
    for model, name in zip(models, model_names):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        results[name] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        print(f"\n{name} 模型性能:")
        print(f"准确率: {results[name]['accuracy']:.4f}")
        print(f"精确率: {results[name]['precision']:.4f}")
        print(f"召回率: {results[name]['recall']:.4f}")
        print(f"F1分数: {results[name]['f1']:.4f}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y, y_pred, f"{name}_confusion_matrix")
    
    return results

def main():
    # 创建必要的目录
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 加载数据
    train_df = load_data('data/processed/train.csv')
    val_df = load_data('data/processed/val.csv')
    
    # 特征工程
    train_df = advanced_feature_engineering(train_df)
    val_df = advanced_feature_engineering(val_df)
    
    # 准备特征和标签
    feature_cols = [col for col in train_df.columns if col not in ['user_log_acct', 'item_sku_id', 'action_time', 'action_type']]
    categorical_features = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'province', 'city', 'county', 'hour_category']
    
    # 将分类特征转换为数值类型
    for col in categorical_features:
        if col in train_df.columns:
            # 使用LabelEncoder将分类特征转换为数值
            le = LabelEncoder()
            # 合并训练集和验证集以确保编码一致性
            combined = pd.concat([train_df[col], val_df[col]])
            le.fit(combined)
            train_df[col] = le.transform(train_df[col])
            val_df[col] = le.transform(val_df[col])
    
    X_train = train_df[feature_cols]
    y_train = train_df['action_type'] - 1  # 将标签转换为0开始
    X_val = val_df[feature_cols]
    y_val = val_df['action_type'] - 1
    
    # 绘制类别分布
    plot_class_distribution(y_train, "训练集类别分布")
    plot_class_distribution(y_val, "验证集类别分布")
    
    # 训练模型
    models = train_models(X_train, y_train, X_val, y_val, feature_cols, categorical_features)
    model_names = ['XGBoost', 'CatBoost', 'LightGBM', 'Voting']
    
    # 评估模型
    results = evaluate_models(models, X_val, y_val, model_names)
    
    # 绘制特征重要性和学习曲线
    plot_feature_importance(models[:-1], feature_cols, model_names[:-1])  # 不包含投票模型
    plot_learning_curves(models[:-1], X_train, y_train, X_val, y_val, model_names[:-1])
    
    # 保存模型
    for model, name in zip(models, model_names):
        joblib.dump(model, f'models/{name.lower()}_model.joblib')
    
    print("\n训练完成！所有模型和图表已保存。")

if __name__ == "__main__":
    main() 