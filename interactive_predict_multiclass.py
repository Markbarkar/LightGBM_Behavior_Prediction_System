#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LightGBM算法的用户行为预测模型 - 多分类交互式预测脚本

该脚本用于加载训练好的LightGBM模型，并根据用户输入的信息进行交互式预测。
模型预测的是用户可能的行为类型（多分类问题），包括浏览、购买、收藏、加购物车等多种行为。

注意：
1. 该脚本不需要用户输入行为类型，因为行为类型正是我们要预测的目标
2. 样本数据的导入是必要的，用于获取特征编码映射和统计信息，确保与训练时的处理一致
3. 该脚本假设已经有一个训练好的多分类模型，如果没有，需要先训练一个多分类模型
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


# 定义行为类型映射字典
ACTION_TYPE_MAPPING = {
    1: "浏览",
    2: "购买",
    3: "收藏",
    4: "加购物车",
    5: "其他行为"
}


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='LightGBM用户行为类型多分类交互式预测')
    parser.add_argument('--model', type=str, default='model/lightgbm_user_behavior_multiclass_model.model', 
                        help='多分类模型文件路径')
    parser.add_argument('--sample_data', type=str, default='./data/processed/val.csv', 
                        help='样本数据文件路径，用于获取特征信息')
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
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在!")
        print("提示: 您可能需要先训练一个多分类模型。")
        print("      当前脚本假设您已经有一个训练好的多分类模型。")
        print("      如果没有，请先使用train_model_multiclass.py训练一个多分类模型。")
        print("\n为了演示，将使用现有的二分类模型进行预测，但只能预测购买/不购买行为。")
        # 使用现有的二分类模型作为备选
        model_path = 'model/lightgbm_user_behavior_model_split.model'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"备选模型文件 {model_path} 也不存在!")
    
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
    
    # 不需要输入行为类型，因为行为类型是我们要预测的目标
    
    return user_input


def predict_user_behavior_multiclass(model, X):
    """
    预测用户可能的行为类型
    
    Args:
        model: 加载的模型
        X: 特征数据
        
    Returns:
        预测结果和预测概率：
        - y_pred: 预测的行为类型（1-5之间的整数）
        - y_pred_proba: 各行为类型的预测概率
    """
    # 检查模型是否支持多分类预测
    num_class = model.num_class() if hasattr(model, 'num_class') else 1
    
    # 预测概率
    if num_class > 1:
        # 多分类模型，返回每个类别的概率
        y_pred_proba = model.predict(X)
        if len(y_pred_proba.shape) == 1:  # 如果只有一个样本，reshape一下
            y_pred_proba = y_pred_proba.reshape(1, -1)
        
        # 获取概率最大的类别作为预测结果
        y_pred = np.argmax(y_pred_proba[0]) + 1  # +1是因为类别从1开始
        
        return y_pred, y_pred_proba[0]
    else:
        # 二分类模型，只能预测是否购买
        y_pred_proba_binary = model.predict(X)[0]  # 获取第一个样本的预测概率
        
        # 转换为二分类结果
        y_pred_binary = 2 if y_pred_proba_binary > 0.5 else 1  # 2表示购买，1表示浏览
        
        # 创建一个5类的概率分布（大部分概率集中在预测的类别上）
        y_pred_proba = np.zeros(5)
        if y_pred_binary == 2:  # 预测为购买
            y_pred_proba[1] = y_pred_proba_binary  # 购买的概率
            y_pred_proba[0] = 1 - y_pred_proba_binary  # 浏览的概率
        else:  # 预测为浏览
            y_pred_proba[0] = 1 - y_pred_proba_binary  # 浏览的概率
            y_pred_proba[1] = y_pred_proba_binary  # 购买的概率
        
        return y_pred_binary, y_pred_proba


def print_prediction_result_multiclass(user_input, y_pred, y_pred_proba):
    """
    打印多分类预测结果
    
    该函数将模型的预测结果以易于理解的方式呈现给用户。
    模型预测的是用户可能的行为类型（多分类问题）。
    
    Args:
        user_input: 用户输入的数据
        y_pred: 预测的行为类型（1-5之间的整数）
        y_pred_proba: 各行为类型的预测概率
    """
    print("\n预测结果:")
    print(f"用户ID: {user_input['user_log_acct']}")
    print(f"商品ID: {user_input['item_sku_id']}")
    print(f"行为时间: {user_input['action_time']}")
    
    # 预测用户行为类型
    predicted_action = ACTION_TYPE_MAPPING.get(y_pred, "未知行为")
    print(f"\n预测行为: 用户很可能会【{predicted_action}】该商品")
    
    # 显示各行为类型的概率
    print("\n各行为类型的概率:")
    
    # 检查y_pred_proba的形状，确保它是一个数组
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        y_pred_proba = [y_pred_proba]  # 如果是单个值，转换为列表
    
    # 如果概率数组长度小于行为类型数量，进行填充
    if len(y_pred_proba) < len(ACTION_TYPE_MAPPING):
        y_pred_proba = list(y_pred_proba) + [0] * (len(ACTION_TYPE_MAPPING) - len(y_pred_proba))
    
    # 打印每种行为类型的概率
    for action_id, action_name in ACTION_TYPE_MAPPING.items():
        if action_id <= len(y_pred_proba):
            prob_index = action_id - 1  # 概率数组索引从0开始
            print(f"  {action_name}: {y_pred_proba[prob_index]:.2%}")
    
    # 根据最高概率给出置信度评估
    max_prob = max(y_pred_proba)
    print("\n预测置信度: ", end="")
    if max_prob > 0.8:
        print("非常高")
    elif max_prob > 0.6:
        print("高")
    elif max_prob > 0.4:
        print("中等")
    else:
        print("低")
    
    # 行为解释
    print("\n关于预测结果的说明:")
    print("- 该模型预测用户可能的行为类型，包括浏览、购买、收藏、加购物车等")
    print("- 预测结果基于用户的历史行为、商品特征和上下文信息")
    print(f"- 最终预测：用户很可能会【{predicted_action}】该商品")
    
    # 如果是使用二分类模型进行的预测，添加说明
    if len(set(y_pred_proba)) <= 2:
        print("\n注意: 当前使用的是二分类模型进行预测，只能区分购买和非购买行为。")
        print("      如需更精确的多分类预测，请训练专门的多分类模型。")


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
    
    print("\n欢迎使用用户行为类型预测系统！")
    print("该系统可以预测用户对商品可能采取的行为类型，包括浏览、购买、收藏、加购物车等。")
    
    while True:
        # 获取用户输入
        user_input = get_user_input(feature_info)
        
        # 预处理用户输入
        X = preprocess_user_input(user_input, feature_cols, sample_df)
        
        # 预测用户行为
        y_pred, y_pred_proba = predict_user_behavior_multiclass(model, X)
        
        # 打印预测结果
        print_prediction_result_multiclass(user_input, y_pred, y_pred_proba)
        
        # 询问是否继续
        continue_input = input("\n是否继续预测？(y/n): ")
        if continue_input.lower() != 'y':
            break
    
    print("预测完成！")


if __name__ == "__main__":
    main()