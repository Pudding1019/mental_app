import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Suicide Risk Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("suicide_risk_model.joblib")

model_data = load_model()

st.title("🧠 Suicide Risk Prediction App")

st.markdown("预测国家或地区的自杀风险等级，请输入以下指标：")

with st.form("input_form"):
    AlcoholUseDisorders = st.slider("Alcohol Use Disorders (%)", 0.0, 15.0, 5.0)
    Unemployment = st.slider("Unemployment Rate (%)", 0.0, 30.0, 10.0)
    Adolescent_Dropout = st.slider("Adolescent Dropout Rate (%)", 0.0, 50.0, 15.0)

    # 原始 Mental Health 三项输入
    BipolarDisorder = st.slider("Bipolar Disorder (%)", 0.0, 10.0, 2.0)
    AnxietyDisorders = st.slider("Anxiety Disorders (%)", 0.0, 20.0, 5.0)
    EatingDisorders = st.slider("Eating Disorders (%)", 0.0, 5.0, 1.0)

    GDP_per_Worker = st.number_input("GDP per Worker", min_value=5000.0, max_value=100000.0, value=40000.0)
    Psychiatrists = st.slider("Psychiatrists (per 10,000)", 0.0, 10.0, 2.0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # 构造 mental health 数据再做 PCA
    mh_df = pd.DataFrame({
        'BipolarDisorder': [BipolarDisorder],
        'AnxietyDisorders': [AnxietyDisorders],
        'EatingDisorders': [EatingDisorders]
    })

    mh_scaled = model_data['mh_scaler'].transform(mh_df)
    MentalHealth_PC1 = model_data['pca'].transform(mh_scaled)[0][0]

    input_df = pd.DataFrame({
        'AlcoholUseDisorders': [AlcoholUseDisorders],
        'Unemployment': [Unemployment],
        'Adolescent_Dropout': [Adolescent_Dropout],
        'MentalHealth_PC1': [MentalHealth_PC1],
        'GDP_per_Worker': [GDP_per_Worker],
        'Psychiatrists(per 10 000 population)': [Psychiatrists]
    })

    # 交互项
    input_df['EcoMental_Interaction'] = input_df['Unemployment'] * input_df['MentalHealth_PC1']
    input_df['Healthcare_Interaction'] = input_df['Psychiatrists(per 10 000 population)'] * input_df['AlcoholUseDisorders']

    # 标准化和预测
    X_input = input_df[model_data['final_features']]
    X_scaled = model_data['scaler'].transform(X_input)
    pred_value = model_data['model'].predict(X_scaled)[0]
    risk_level = pd.cut(
        [pred_value],
        bins=model_data['risk_bins'],
        labels=model_data['risk_labels'],
        include_lowest=True
    )[0]

    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Risk Value:** `{pred_value:.2f}`")
    st.write(f"**Risk Level:** :red[{risk_level}]")

    fig, ax = plt.subplots()
    ax.bar(model_data['risk_labels'], [int(risk_level == label) for label in model_data['risk_labels']])
    ax.set_ylabel("Confidence (1 = predicted category)")
    ax.set_title("Risk Category")
    st.pyplot(fig)
