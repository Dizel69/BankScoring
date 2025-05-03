"""
Скрипт для обучения финальных моделей: XGBoost, LightGBM и нейронной сети (MLPClassifier).
Данные ожидаются в формате CSV (например, ранее обработанный набор данных,
такой как poland_processed.csv или любой другой вариант).

После обучения модели сохраняются в файлы для дальнейшего использования в финальном приложении:
- xgb_model.pkl
- lgb_model.pkl
- nn_model.pkl
- scaler.pkl

Запустите этот скрипт для обучения и сохранения моделей.
"""

import pandas as pd
import numpy as np
import joblib  # Для сохранения моделей
import os

# Импортируем необходимые библиотеки для моделей
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Импорт моделей
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Параметры можно настроить в соответствии с особенностями ваших данных
DATA_PATH = 'verdata/poland_processed.csv'  # Замените на нужный путь к вашему файлу


def load_and_preprocess_data(file_path):
    """
    Функция загружает данные из CSV и разделяет их на признаки (X) и целевую переменную (y).
    """
    df = pd.read_csv(file_path)
    # Проверяем, что столбец 'target' присутствует
    if 'target' not in df.columns:
        raise ValueError("В данных отсутствует столбец 'target'.")
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def train_and_evaluate_models(X, y):
    """
    Функция разделяет данные на обучающую и тестовую выборки, масштабирует признаки,
    обучает модели XGBoost, LightGBM и NeuralNetwork, рассчитывает метрики на тестовой выборке,
    и возвращает обученные модели и масштабатор.
    """
    # Разделение данных: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирование данных (важно для нейронной сети и может помочь для бустинга)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели XGBoost
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    # Для XGBoost используем необязательное масштабирование, но можно использовать исходные данные:
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Обучение модели LightGBM
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # Обучение нейронной сети (MLPClassifier)
    # Используем масштабированные данные для нейронной сети
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)
    y_proba_nn = nn_model.predict_proba(X_test_scaled)[:, 1]

    # Рассчитываем и выводим основные метрики для каждой модели
    def print_metrics(model_name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        print(f'=== {model_name} ===')
        print(f'Точность (Accuracy): {acc:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'ROC-AUC: {auc:.4f}')
        print(classification_report(y_true, y_pred))
        print('-------------------------')

    print_metrics("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
    print_metrics("LightGBM", y_test, y_pred_lgb, y_proba_lgb)
    print_metrics("NeuralNetwork", y_test, y_pred_nn, y_proba_nn)

    return xgb_model, lgb_model, nn_model, scaler


def save_trained_models(xgb_model, lgb_model, nn_model, scaler, output_dir='final_models'):
    """
    Сохраняет обученные модели и масштабатор в указанный каталог.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(xgb_model, os.path.join(output_dir, 'xgb_model.pkl'))
    joblib.dump(lgb_model, os.path.join(output_dir, 'lgb_model.pkl'))
    joblib.dump(nn_model, os.path.join(output_dir, 'nn_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("Обученные модели и масштабатор успешно сохранены в каталоге:", output_dir)


def main():
    # Загружаем данные
    print("Загрузка данных из файла:", DATA_PATH)
    X, y = load_and_preprocess_data(DATA_PATH)

    # Обучаем модели и оцениваем их
    xgb_model, lgb_model, nn_model, scaler = train_and_evaluate_models(X, y)

    # Сохраняем обученные модели и scaler
    save_trained_models(xgb_model, lgb_model, nn_model, scaler)


if __name__ == "__main__":
    main()
