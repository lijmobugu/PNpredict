# 新增必要的导入
import shap
from sklearn.preprocessing import LabelEncoder

# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习库
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer

# 分类算法（保持与训练时一致）
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
import streamlit as st

# 加载模型和标准化器
try:
    model = joblib.load('stacking_classifier.pkl')
    scaler = joblib.load('scaler.pkl')  # 新增标准化器加载
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please ensure both model and scaler files are uploaded.")
    st.stop()

# 特征范围定义（确保顺序与训练数据一致）
feature_names = [
    "age", "cm", "ASA score", "Smoke", "Drink","Fever", 
    "linbaxibaojishu", "HB", "PLT","ningxuemeiyuanshijian"
]

feature_ranges = {
    "age": {"type": "numerical", "min": 18, "max": 80, "default": 40},
    "cm": {"type": "numerical", "min": 140, "max": 170, "default": 160},
    "ASA score": {"type": "categorical", "options": ["1", "2", "3"]},
    "Smoke": {"type": "categorical", "options": ["YES", "NO"]},
    "Drink": {"type": "categorical", "options": ["YES", "NO"]},
    "Fever": {"type": "categorical", "options": ["YES", "NO"]},
    "linbaxibaojishu": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "HB": {"type": "numerical", "min": 0, "max": 200, "default": 0},
    "PLT": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "ningxuemeiyuanshijian": {"type": "numerical", "min": 0, "max": 170, "default": 0}
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 特征输入处理
feature_values = {}
for feature in feature_names:  # 按照指定顺序遍历
    properties = feature_ranges[feature]
    
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 分类特征编码
label_encoders = {}
for feature in feature_names:
    if feature_ranges[feature]["type"] == "categorical":
        le = LabelEncoder()
        le.fit(feature_ranges[feature]["options"])
        feature_values[feature] = le.transform([feature_values[feature]])

# 创建DataFrame并标准化
try:
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    features_scaled = scaler.transform(features_df)  # 应用标准化
except ValueError as e:
    st.error(f"Feature processing error: {e}. Check feature names and order.")
    st.stop()

# 预测与可视化
if st.button("Predict"):
    try:
        # 模型预测
        proba = model.predict_proba(features_scaled)  # 获取阳性概率
        st.subheader("Prediction Result:")
        st.write(f"Predicted possibility of AKI is **{proba*100:.2f}%**")

        # SHAP解释（使用KernelExplainer）
        background = shap.sample(features_scaled, 10)  # 使用少量样本作为背景
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(features_scaled)
        
        # 生成并显示力图
        plt.figure()
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=features_scaled,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
