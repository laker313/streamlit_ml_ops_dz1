import streamlit as st
import requests
import pandas as pd
import io
import json
from app.models.models import Models, MODEL_CLASSES, Model_Type

st.set_page_config(page_title="MLOps Dashboard", layout="wide")

st.title("🎯 MLOps Dashboard")
# st.sidebar.header("Управление моделями")
api_url = st.sidebar.text_input("API URL", value="http://localhost:80")


# Основные вкладки
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Создание моделей", 
    "🎓 Обучение", 
    "🔮 Предсказания", 
    "📋 Информация", 
    "⚙️ Управление", 
    "📈 Мониторинг",
    "🗃️ Управление датасетами"
])

# Вкладка 1: Создание моделей
with tab1:
    st.header("Создание моделей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Быстрое создание")
        model_name = st.selectbox("Модель", list(MODEL_CLASSES.keys()))
        task_type = st.selectbox("Тип задачи", [Model_Type.CLASSIFIER.value, Model_Type.REGRESSOR.value])
        
        if st.button("Создать модель (стандартные параметры)"):
            try:
                response = requests.post(
                    f"{api_url}/api/v1/models/create_and_save_model",
                    data={"model_name": model_name, "task_type": task_type, "hyperparameters": "{}"}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Модель создана!")
                    st.json(result)
                else:
                    st.error(f"❌ Ошибка: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка подключения: {e}")
    
    with col2:
        st.subheader("Расширенное создание")
        advanced_model_name = st.selectbox("Модель (расш.)", list(MODEL_CLASSES.keys()), key="advanced_model")
        advanced_task_type = st.selectbox("Тип задачи (расш.)", [Model_Type.CLASSIFIER.value, Model_Type.REGRESSOR.value], key="advanced_task")
        
        hyperparams = st.text_area("Гиперпараметры (JSON)", value='{"n_estimators": 100, "random_state": 42}')
        
        if st.button("Создать модель (с гиперпараметрами)"):
            try:
                # Валидация JSON
                json.loads(hyperparams)
                response = requests.post(
                    f"{api_url}/api/v1/models/create_and_save_model",
                    data={
                        "model_name": advanced_model_name, 
                        "task_type": advanced_task_type, 
                        "hyperparameters": hyperparams
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Модель создана с гиперпараметрами!")
                    st.json(result)
                else:
                    st.error(f"❌ Ошибка: {response.text}")
            except json.JSONDecodeError:
                st.error("❌ Невалидный JSON в гиперпараметрах")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Вкладка 2: Обучение моделей
with tab2:
    st.header("Обучение моделей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Обучение модели")
        train_model_id = st.text_input("ID модели для обучения", key="train_model_id")
        train_data_id = st.text_input("ID датасета для обучения", key="train_data_id")
        
        if st.button("Обучить модель"):
            if train_model_id and train_data_id:
                try:
                    with st.spinner("Обучаем модель..."):
                        response = requests.post(
                            f"{api_url}/api/v1/models/learn_model",
                            data={"model_id": train_model_id, "data_id": train_data_id}
                        )
                        if response.status_code == 200:
                            st.success("✅ Модель успешно обучена!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ Ошибка обучения: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
            else:
                st.warning("⚠️ Введите ID модели и датасета")
    
    with col2:
        st.subheader("Обновление модели")
        update_model_name = st.selectbox("Модель для обновления", list(MODEL_CLASSES.keys()), key="update_model")
        update_task_type = st.selectbox("Тип задачи для обновления", [Model_Type.CLASSIFIER.value, Model_Type.REGRESSOR.value], key="update_task")
        update_hyperparams = st.text_area("Новые гиперпараметры", value='{"n_estimators": 150}', key="update_params")
        
        if st.button("Обновить модель"):
            try:
                json.loads(update_hyperparams)
                response = requests.post(
                    f"{api_url}/api/v1/models/update_model",
                    data={
                        "model_name": update_model_name,
                        "task_type": update_task_type,
                        "hyperparameters": update_hyperparams
                    }
                )
                if response.status_code == 200:
                    st.success("✅ Модель обновлена!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Ошибка: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Вкладка 3: Предсказания
with tab3:
    st.header("Получение предсказаний")
    
    pred_model_id = st.text_input("ID обученной модели", key="pred_model_id")
    pred_file = st.file_uploader("Загрузите данные для предсказаний", type=['csv', 'parquet'], key="pred_file")
    
    if st.button("Получить предсказания"):
        if pred_model_id and pred_file:
            try:
                with st.spinner("Получаем предсказания..."):
                    files = {"file": (pred_file.name, pred_file.getvalue(), pred_file.type)}
                    response = requests.post(
                        f"{api_url}/api/v1/models/get_predictions_from_file",
                        data={"model_id": pred_model_id},
                        files=files
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Предсказания получены!")
                        
                        # Скачивание результатов
                        predictions_df = pd.read_csv(io.BytesIO(response.content), sep=None, engine='python')
                        st.dataframe(predictions_df)
                        
                        # Кнопка скачивания
                        csv = predictions_df.to_csv()
                        st.download_button(
                            label="📥 Скачать предсказания",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"❌ Ошибка: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
        else:
            st.warning("⚠️ Введите ID модели и загрузите файл")

# Вкладка 4: Информация о моделях
with tab4:
    st.header("Информация о моделях")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Получить информацию о модели")
        info_model_id = st.text_input("ID модели для информации", key="info_model_id")
        
        if st.button("Получить информацию"):
            if info_model_id:
                try:
                    response = requests.post(
                        f"{api_url}/api/v1/models/get_model",
                        data={"model_id": info_model_id}
                    )
                    if response.status_code == 200:
                        info = response.json()
                        st.success("✅ Информация получена!")
                        
                        st.metric("Название модели", info["model_name"])
                        st.metric("Статус обучения", info["learning_status"])
                        st.json(info["hyperparams"])
                    else:
                        st.error(f"❌ Ошибка: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
    
    with col2:
        st.subheader("Список доступных моделей")
        if st.button("Обновить список моделей"):
            try:
                response = requests.get(f"{api_url}/api/v1/models/type_list")
                if response.status_code == 200:
                    models_list = response.json()["message"]
                    st.write("📋 Доступные модели:")
                    for model in models_list:
                        st.write(f"- {model}")
                else:
                    st.error(f"❌ Ошибка: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Вкладка 5: Управление моделями
with tab5:
    st.header("Управление моделями")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Удаление модели")
        delete_model_id = st.text_input("ID модели для удаления", key="delete_model_id")
        
        if st.button("🗑️ Удалить модель", type="secondary"):
            if delete_model_id:
                try:
                    response = requests.post(
                        f"{api_url}/api/v1/models/delete_model",
                        data={"model_id": delete_model_id}
                    )
                    if response.status_code == 200:
                        st.success("✅ Модель удалена!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Ошибка: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
    
    with col2:
        st.subheader("Гиперпараметры по умолчанию")
        params_model_name = st.selectbox("Модель для параметров", list(MODEL_CLASSES.keys()), key="params_model")
        params_task_type = st.selectbox("Тип задачи для параметров", [Model_Type.CLASSIFIER.value, Model_Type.REGRESSOR.value], key="params_task")
        
        if st.button("Показать параметры по умолчанию"):
            try:
                # Используем локальную функцию
                from app.models.models import get_model_default_params
                params = get_model_default_params(params_model_name, params_task_type)
                st.json(params)
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Вкладка 6: Мониторинг
with tab6:
    st.header("Мониторинг системы")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Статус здоровья")
        if st.button("Проверить здоровье системы"):
            try:
                response = requests.get(f"{api_url}/api/v1/models/health")
                if response.status_code == 200:
                    health = response.json()
                    st.success("✅ Система работает нормально")
                    st.metric("Статус", health["status"])
                    st.metric("Рабочие потоки", health["workers"])
                    st.metric("Размер очереди", health["queue_size"])
                else:
                    st.error("❌ Проблемы с системой")
            except Exception as e:
                st.error(f"❌ Ошибка подключения: {e}")
    
    with col2:
        st.subheader("Статус пула потоков")
        if st.button("Проверить пул потоков"):
            try:
                response = requests.get(f"{api_url}/api/v1/models/pool_status")
                if response.status_code == 200:
                    pool_status = response.json()
                    st.metric("Макс. потоков", pool_status["max_workers"])
                    st.metric("Активных потоков", pool_status["active"])
                    st.metric("Задач в очереди", pool_status["queue"])
                else:
                    st.error("❌ Ошибка получения статуса пула")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

with tab7:
    st.header("🗃️ Управление датасетами")
    
    # Подвкладки для датасетов
    dataset_tab1, dataset_tab2, dataset_tab3, dataset_tab4 = st.tabs([
        "📤 Загрузка датасетов",
        "🔄 Обновление датасетов", 
        "📥 Скачивание датасетов",
        "🗑️ Удаление датасетов"
    ])
    
    # Подвкладка 1: Загрузка датасетов
    with dataset_tab1:
        st.subheader("Загрузка датасетов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Загрузить новый датасет**")
            upload_file = st.file_uploader("Выберите файл датасета", 
                                         type=['csv', 'parquet', 'json'], 
                                         key="dataset_upload")
            
            if upload_file and st.button("📤 Загрузить датасет"):
                try:
                    files = {"file": (upload_file.name, upload_file.getvalue(), upload_file.type)}
                    
                    with st.spinner("Загружаем датасет..."):
                        response = requests.post(
                            f"{api_url}/api/v1/data/upload_dataset",
                            files=files
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Датасет успешно загружен!")
                            st.metric("ID датасета", result["dataset_id"])
                            st.metric("Название", result["dataset_name"])
                            st.json(result)
                        else:
                            st.error(f"❌ Ошибка загрузки: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
        
        with col2:
            st.write("**Информация о поддерживаемых форматах**")
            st.info("""
            **Поддерживаемые форматы:**
            - 📁 CSV (.csv)
            - 📁 Parquet (.parquet) 
            - 📁 JSON (.json)
            - 📁 Pickle (.pkl, .pickle)
            - 📁 Feather (.feather)
            
            **Требования:**
            - Должен содержать столбец 'target'
            - Только один столбец 'target'
            - Без пропущенных значений в 'target'
            """)
    
    # Подвкладка 2: Обновление датасетов
    with dataset_tab2:
        st.subheader("Обновление датасетов")
        
        update_dataset_id = st.text_input("ID датасета для обновления", key="update_dataset_id")
        update_file = st.file_uploader("Выберите новый файл датасета", 
                                     type=['csv', 'parquet', 'json'],
                                     key="update_dataset_file")
        
        if st.button("🔄 Обновить датасет"):
            if update_dataset_id and update_file:
                try:
                    files = {"file": (update_file.name, update_file.getvalue(), update_file.type)}
                    data = {"dataset_id": update_dataset_id}
                    
                    with st.spinner("Обновляем датасет..."):
                        response = requests.post(
                            f"{api_url}/api/v1/data/update_dataset",
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Датасет успешно обновлен!")
                            st.metric("ID датасета", result["dataset_id"])
                            st.metric("Название", result["dataset_name"])
                            st.json(result)
                        else:
                            st.error(f"❌ Ошибка обновления: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
            else:
                st.warning("⚠️ Введите ID датасета и загрузите файл")
    
    # Подвкладка 3: Скачивание датасетов
    with dataset_tab3:
        st.subheader("Скачивание датасетов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Скачать датасет**")
            download_dataset_id = st.text_input("ID датасета для скачивания", key="download_dataset_id")
            
            if st.button("📥 Скачать датасет (CSV)"):
                if download_dataset_id:
                    try:
                        with st.spinner("Скачиваем датасет..."):
                            response = requests.post(
                                f"{api_url}/api/v1/data/download_dataset",
                                data={"dataset_id": download_dataset_id}
                            )
                            
                            if response.status_code == 200:
                                st.success("✅ Датасет скачан!")
                                
                                # Показать предпросмотр
                                dataset_df = pd.read_csv(io.BytesIO(response.content))
                                st.write(f"**Предпросмотр датасета ({len(dataset_df)} строк):**")
                                st.dataframe(dataset_df.head(10))
                                
                                # Кнопка скачивания
                                csv = dataset_df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Скачать CSV",
                                    data=csv,
                                    file_name=f"dataset_{download_dataset_id}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error(f"❌ Ошибка скачивания: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
                else:
                    st.warning("⚠️ Введите ID датасета")
        
        with col2:
            st.write("**Быстрый просмотр**")
            quick_dataset_id = st.text_input("ID датасета для быстрого просмотра", key="quick_view_id")
            
            if st.button("👀 Быстрый просмотр"):
                if quick_dataset_id:
                    try:
                        # Используем тот же эндпоинт для предпросмотра
                        response = requests.post(
                            f"{api_url}/api/v1/data/download_dataset",
                            data={"dataset_id": quick_dataset_id}
                        )
                        
                        if response.status_code == 200:
                            dataset_df = pd.read_csv(io.BytesIO(response.content))
                            
                            st.metric("Строки", len(dataset_df))
                            st.metric("Столбцы", len(dataset_df.columns))
                            st.metric("Размер", f"{len(response.content) / 1024:.1f} KB")
                            
                            st.write("**Столбцы:**")
                            for col in dataset_df.columns:
                                st.write(f"- {col}")
                        else:
                            st.error(f"❌ Ошибка: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
    
    # Подвкладка 4: Удаление датасетов
    with dataset_tab4:
        st.subheader("Удаление датасетов")
        
        st.warning("⚠️ Внимание: Удаление датасета необратимо!")
        
        delete_dataset_id = st.text_input("ID датасета для удаления", key="delete_dataset_id")
        
        # Подтверждение удаления
        if delete_dataset_id:
            confirm_delete = st.checkbox("Я понимаю, что это действие необратимо")
            
            if confirm_delete and st.button("🗑️ Удалить датасет", type="secondary"):
                try:
                    response = requests.post(
                        f"{api_url}/api/v1/data/delete_dataset",
                        data={"dataset_id": delete_dataset_id}
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Датасет удален!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Ошибка удаления: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

# Боковая панель - быстрый доступ к датасетам
st.sidebar.header("🗃️ Быстрый доступ к датасетам")

# Загрузка датасета через боковую панель
sidebar_upload_file = st.sidebar.file_uploader("Быстрая загрузка датасета", 
                                             type=['csv', 'parquet'],
                                             key="sidebar_upload")

if sidebar_upload_file and st.sidebar.button("🚀 Быстрая загрузка"):
    try:
        files = {"file": (sidebar_upload_file.name, sidebar_upload_file.getvalue(), sidebar_upload_file.type)}
        
        with st.spinner("Загружаем..."):
            response = requests.post(
                f"{api_url}/api/v1/data/upload_dataset",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(f"✅ Загружен: {result['dataset_id']}")
            else:
                st.sidebar.error("❌ Ошибка загрузки")
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка: {e}")

# Информация в боковой панели
st.sidebar.header("ℹ️ Справка по датасетам")
st.sidebar.info("""
**Для работы с ML:**
- Датасеты хранятся в MinIO
- Автоматическая валидация формата
- Обязателен столбец 'target'
- Поддержка multiple форматов
""")

# Обновление страницы
if st.sidebar.button("🔄 Обновить все статусы"):
    st.rerun()