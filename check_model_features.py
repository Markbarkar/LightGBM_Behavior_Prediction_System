#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查看LightGBM模型的特征名称
"""

import lightgbm as lgb

# 加载模型
model = lgb.Booster(model_file='model/lightgbm_user_behavior_model.txt')

# 打印特征数量
print('特征数量:', len(model.feature_name()))

# 打印特征名称
print('特征名称:', model.feature_name())