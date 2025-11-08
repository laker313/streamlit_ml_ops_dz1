import streamlit as st
import requests
import pandas as pd
import io
from app.models.models import Models, MODEL_CLASSES
st.set_page_config(page_title="MLOps Dashboard", layout="wide")

st.title("MLOps Dashboard")
st.sidebar.header("Управление моделями")

# Загрузка датасета
uploaded_file = st.sidebar.file_uploader("Загрузите датасет", type=['csv', 'parquet'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
    st.write("Датасет:", df.shape)
    st.dataframe(df.head())

# Создание модели
st.sidebar.subheader("Создать модель")
model_name = st.sidebar.selectbox("Модель", MODEL_CLASSES.keys())
task_type = st.sidebar.selectbox("Тип задачи", ["classification", "regression"])

if st.sidebar.button("Создать модель"):
    response = requests.post(
        "http://localhost:80/api/v1/models/create_and_save_model",
        data={"model_name": model_name, "task_type": task_type}
    )
    if response.status_code == 200:
        st.sidebar.success(f"Модель создана: {response.json()['model_id']}")

# Обучение модели
st.sidebar.subheader("Обучение")
model_id = st.sidebar.text_input("ID модели")
data_id = st.sidebar.text_input("ID датасета")

if st.sidebar.button("Обучить модель"):
    response = requests.post(
        "http://localhost:80/api/v1/models/learn_model",
        data={"model_id": model_id, "data_id": data_id}
    )
    st.sidebar.json(response.json())