#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 交互式预测脚本

该脚本用于加载训练好的LightGBM模型，并根据用户输入的信息进行交互式预测。
模型预测的是用户是否会购买商品（二分类问题），而不是预测具体的行为类型。

注意：
1. 该脚本不需要用户输入行为类型，因为行为类型（是否购买）正是我们要预测的目标
2. 样本数据的导入是必要的，用于获取特征编码映射和统计信息，确保与训练时的处理一致
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import argparse
import time
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为预测交互式预测')
    parser.add_argument('--model', type=str, default='model/lightgbm_user_behavior_model_split.model', help='模型文件路径')
    parser.add_argument('--sample_data', type=str, default='./data/processed/val.csv', help='样本数据文件路径，用于获取特征信息')
    return parser.parse_args()


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


def load_sample_data(file_path):
    """
    加载样本数据，用于获取特征信息
    
    注意：样本数据的导入是必要的，因为：
    1. 需要获取分类特征的编码映射，确保与训练时一致
    2. 需要获取数值特征的统计信息，用于填充缺失值
    3. 需要了解特征的分布情况，以便进行合理的预处理
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        DataFrame: 加载的数据集
    """
    print(f"加载样本数据: {file_path}")
    print("(样本数据仅用于特征编码和统计信息，不会影响预测结果的客观性)")
    df = pd.read_csv(file_path)
    return df


def get_feature_info(df):
    """
    获取特征信息，包括特征名称、类型、取值范围等
    
    Args:
        df: 样本数据集
        
    Returns:
        特征信息字典
    """
    feature_info = {}
    
    # 获取数值型特征的统计信息
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        feature_info[col] = {
            'type': 'numeric',
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median()
        }
    
    # 获取分类特征的取值
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        feature_info[col] = {
            'type': 'categorical',
            'values': df[col].unique().tolist()
        }
    
    return feature_info


def preprocess_user_input(user_input, feature_cols, sample_df):
    """
    预处理用户输入的数据
    
    Args:
        user_input: 用户输入的数据字典
        feature_cols: 模型所需的特征列
        sample_df: 样本数据，用于特征编码
        
    Returns:
        处理后的特征数据
    """
    # 创建一个包含用户输入的DataFrame
    input_df = pd.DataFrame([user_input])
    
    # 处理时间特征
    if 'action_time' in input_df.columns:
        input_df['action_time'] = pd.to_datetime(input_df['action_time'])
        input_df['hour'] = input_df['action_time'].dt.hour
        input_df['day'] = input_df['action_time'].dt.day
        input_df['month'] = input_df['action_time'].dt.month
        input_df['dayofweek'] = input_df['action_time'].dt.dayofweek
    
    # 处理缺失值
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].fillna('unknown')
        else:
            input_df[col] = input_df[col].fillna(-1)
    
    # 对分类特征进行编码
    categorical_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'province', 'city', 'county']
    for col in categorical_cols:
        if col in input_df.columns and col in sample_df.columns:
            # 使用样本数据中的唯一值作为参考进行编码
            unique_values = sample_df[col].astype(str).unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            
            # 如果用户输入的值不在映射中，则设为-1
            input_df[col] = input_df[col].astype(str).map(lambda x: mapping.get(x, -1))
    
    # 确保所有需要的特征列都存在
    # 这里我们直接使用模型需要的特征列表，而不是动态生成
    required_features = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'shop_score', 
                        'age', 'sex', 'user_level', 'province', 'city', 'county', 'hour', 'day', 
                        'month', 'dayofweek', 'user_item_count', 'user_action_mean', 'user_shop_nunique', 
                        'user_brand_nunique', 'user_cate_nunique', 'item_user_count', 'item_action_mean']
    
    # 为缺失的特征添加默认值
    for col in required_features:
        if col not in input_df.columns:
            if col in sample_df.columns:
                if sample_df[col].dtype in ['int64', 'float64']:
                    input_df[col] = sample_df[col].mean()
                else:
                    input_df[col] = -1
            else:
                input_df[col] = -1
    
    # 创建最终的特征DataFrame，确保列的顺序与模型期望的一致
    X = pd.DataFrame()
    for col in required_features:
        X[col] = input_df[col]
    
    print(f"特征数量: {len(X.columns)}")
    print(f"特征列: {list(X.columns)}")
    
    return X


def get_user_input(feature_info):
    """
    获取用户输入的信息
    
    Args:
        feature_info: 特征信息字典
        
    Returns:
        用户输入的数据字典
    """
    user_input = {}
    
    print("\n请输入用户和商品信息（按Enter跳过将使用默认值）:")
    
    # 必填字段
    user_input['user_log_acct'] = input("用户ID: ") or "unknown"
    user_input['item_sku_id'] = input("商品ID: ") or "unknown"
    user_input['action_time'] = input("行为时间 (YYYY-MM-DD HH:MM:SS): ") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 可选字段
    print("\n以下字段为可选，按Enter跳过将使用默认值或从样本数据中推断:")
    
    # 用户相关特征
    user_input['age'] = input("用户年龄: ") or -1
    if user_input['age'] != -1:
        user_input['age'] = int(user_input['age'])
    
    sex_input = input("用户性别 (0-女性, 1-男性): ")
    user_input['sex'] = int(sex_input) if sex_input else -1
    
    user_level_input = input("用户等级: ")
    user_input['user_level'] = int(user_level_input) if user_level_input else -1
    
    user_input['province'] = input("省份: ") or "unknown"
    user_input['city'] = input("城市: ") or "unknown"
    user_input['county'] = input("区县: ") or "unknown"
    
    # 商品相关特征
    user_input['brand_code'] = input("品牌代码: ") or "unknown"
    user_input['shop_id'] = input("店铺ID: ") or "unknown"
    user_input['item_third_cate_cd'] = input("商品三级类目: ") or "unknown"
    user_input['vender_id'] = input("供应商ID: ") or "unknown"
    
    shop_score_input = input("店铺评分: ")
    user_input['shop_score'] = float(shop_score_input) if shop_score_input else -1
    
    # 不再需要输入行为类型，因为行为类型是我们要预测的目标
    
    return user_input


def predict_user_behavior(model, X):
    """
    预测用户是否会购买商品
    
    Args:
        model: 加载的模型
        X: 特征数据
        
    Returns:
        预测结果和预测概率：
        - y_pred: 1表示预测用户会购买，0表示预测用户不会购买
        - y_pred_proba: 预测用户会购买的概率
    """
    # 预测概率
    y_pred_proba = model.predict(X)[0]  # 获取第一个样本的预测概率
    
    # 转换为二分类结果
    y_pred = 1 if y_pred_proba > 0.5 else 0
    
    return y_pred, y_pred_proba


def print_prediction_result(user_input, y_pred, y_pred_proba):
    """
    打印预测结果
    
    该函数将模型的预测结果以易于理解的方式呈现给用户。
    模型预测的是用户是否会购买商品（二分类问题），而不是预测多种行为类型。
    这是因为在训练模型时，我们将购买行为（action_type=2）作为正样本，
    其他行为（浏览、加购物车、收藏等）作为负样本进行训练。
    
    Args:
        user_input: 用户输入的数据
        y_pred: 预测结果（1表示会购买，0表示不会购买）
        y_pred_proba: 预测用户会购买的概率
    """
    print("\n预测结果:")
    print(f"用户ID: {user_input['user_log_acct']}")
    print(f"商品ID: {user_input['item_sku_id']}")
    print(f"行为时间: {user_input['action_time']}")
    
    # 预测用户行为类型
    if y_pred == 1:
        print(f"预测行为: 用户很可能会【购买】该商品 (概率: {y_pred_proba:.2%})")
    else:
        print(f"预测行为: 用户可能【不会购买】该商品 (概率: {1-y_pred_proba:.2%})")
    
    # 根据概率给出更详细的解释
    print("置信度: ", end="")
    if y_pred_proba > 0.9 or y_pred_proba < 0.1:
        print("非常高")
    elif y_pred_proba > 0.7 or y_pred_proba < 0.3:
        print("高")
    elif y_pred_proba > 0.5 or y_pred_proba < 0.5:
        print("中等")
    
    # 行为解释
    print("\n关于预测结果的说明:")
    print("- 该模型专门预测用户是否会购买商品，而不是预测其他类型的行为")
    print("- 在训练时，购买行为被标记为正样本(1)，其他行为(浏览、加购物车、收藏)被标记为负样本(0)")
    print("- 因此，模型输出的是用户购买该商品的可能性，而不是预测用户会执行哪种具体行为")
    print(f"- 最终预测：用户{'很可能会购买该商品' if y_pred == 1 else '可能不会购买该商品'}")
    print("\n(注：如需预测其他类型的用户行为，需要重新训练针对该行为的模型)")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载模型
    model = load_model(args.model)
    
    # 加载样本数据
    sample_df = load_sample_data(args.sample_data)
    
    # 获取特征信息
    feature_info = get_feature_info(sample_df)
    
    # 获取模型所需的特征列 - 这里使用模型实际需要的特征列表
    feature_cols = ['brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'shop_score', 
                   'age', 'sex', 'user_level', 'province', 'city', 'county', 'hour', 'day', 
                   'month', 'dayofweek', 'user_item_count', 'user_action_mean', 'user_shop_nunique', 
                   'user_brand_nunique', 'user_cate_nunique', 'item_user_count', 'item_action_mean']
    
    while True:
        # 获取用户输入
        user_input = get_user_input(feature_info)
        
        # 预处理用户输入
        X = preprocess_user_input(user_input, feature_cols, sample_df)
        
        # 预测用户行为
        y_pred, y_pred_proba = predict_user_behavior(model, X)
        
        # 打印预测结果
        print_prediction_result(user_input, y_pred, y_pred_proba)
        
        # 询问是否继续
        continue_input = input("\n是否继续预测？(y/n): ")
        if continue_input.lower() != 'y':
            break
    
    print("预测完成！")


if __name__ == "__main__":
    main()