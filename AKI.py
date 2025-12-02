import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('XGBoost.pkl')  # 加载训练好的XGBoost模型

# Define the feature options
ethnicity_options = {
    1: 'White',  
    2: 'Others',  
}

congestive_heart_failure_options = {
    0: 'No',  
    1: 'Yes',  
}

peptic_ulcer_disease_options = {
    0: 'No',  
    1: 'Yes',  
}

Sepsis_options = {
    0: 'No',  
    1: 'Yes',  
}

Invasive_ventilate_options = {
    0: 'No',  
    1: 'Yes',  
}


# Streamlit UI
st.title("Predictors for Acute Kidney Injury in Acute Pancreatitis")  # 预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Weight input
Weight = st.sidebar.number_input("Weight(Kg):", min_value=1, max_value=250, value=80)  # Weight输入框

# ethnicity input
ethnicity = st.sidebar.selectbox("ethnicity:", options=list(ethnicity_options.keys()), format_func=lambda x: ethnicity_options[x])  # 类型选择框

# congestive_heart_failure input
congestive_heart_failure = st.sidebar.selectbox("congestive_heart_failure:", options=list(congestive_heart_failure_options.keys()), format_func=lambda x: congestive_heart_failure_options[x])  # 类型选择框

# peptic_ulcer_disease input
peptic_ulcer_disease = st.sidebar.selectbox("peptic_ulcer_disease:", options=list(peptic_ulcer_disease_options.keys()), format_func=lambda x: peptic_ulcer_disease_options[x])  # 类型选择框

# Sepsis input
Sepsis = st.sidebar.selectbox("Sepsis:", options=list(Sepsis_options.keys()), format_func=lambda x: Sepsis_options[x])  # 类型选择框

# Invasive_ventilate input
Invasive_ventilate = st.sidebar.selectbox("Invasive_ventilate:", options=list(Invasive_ventilate_options.keys()), format_func=lambda x: Invasive_ventilate_options[x])  # 类型选择框

# GCS input
GCS = st.sidebar.number_input("GCS:", min_value=3, max_value=15, value=15)  # 输入框

# OASIS input
OASIS = st.sidebar.number_input("OASIS:", min_value=12, max_value=64, value=30)  # 输入框

# SOFA input
SOFA = st.sidebar.number_input("SOFA:", min_value=0, max_value=12, value=2)  # 输入框

# Respiratory_rate input
Respiratory_rate = st.sidebar.number_input("Respiratory_rate:", min_value=7, max_value=54, value=20)  # 输入框

# SpO2 input
SpO2 = st.sidebar.number_input("SpO2:", min_value=42, max_value=100, value=98)  # 输入框


# WBC input
WBC = st.sidebar.number_input("WBC(10^9/L):", min_value=0.1, max_value=77.8, value=10.0)  # 输入框


# MCHC input
MCHC  = st.sidebar.number_input("MCHC(%):", min_value=25.7, max_value=40.3, value=33.0)  # 输入框

# Total_bilirubin input
Total_bilirubin = st.sidebar.number_input("Total_bilirubin(mg/dL):", min_value=0.1, max_value=41.6, value=17.0)  # 输入框

# Magnesium input
Magnesium = st.sidebar.number_input("Magnesium(mg/dL):", min_value=0.0, max_value=9.6, value=5.0)  # 输入框

# Phosphate input
Phosphate = st.sidebar.number_input("Phosphate(mg/dL):", min_value=0.0, max_value=9.6, value=5.0)  # 输入框

# Process the input and make a prediction
feature_values = [Weight, ethnicity, congestive_heart_failure, peptic_ulcer_disease, Sepsis, Invasive_ventilate, GCS, OASIS, SOFA, Respiratory_rate, SpO2, WBC, MCHC, Total_bilirubin, Magnesium, Phosphate]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为死亡
        advice = (
            f"According to our model, the patient risk of AKI is {probability:.1f}%. "
        )  # 如果预测为aki，给出相关建议
    else:  # 如果预测为non-aki
        advice = (
            f"According to our model, the patient risk of Non AKI is {probability:.1f}%. "
        )  # 如果预测为non-aki，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['Not Sick', 'Sick'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表