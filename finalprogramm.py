"""
Финальное приложение для принятия решения по выдаче кредита (на основе 64 признаков),
которое принимает файл с данными клиента.

Требования:
- Файл должен быть в формате CSV (client_data.csv), содержащим ровно одну строку с 64 числовыми значениями (без заголовка).
- Предобученные модели (xgb_model.pkl, lgb_model.pkl, nn_model.pkl) и масштабатор (scaler.pkl) сохранены в каталоге 'final_models'.

Пример файла client_data.csv:
0.20055,0.37951,0.39641,2.0472,...,(всего 64 числа)
"""

import sys
import joblib
import pandas as pd

def load_models():
    """
    Загружает предобученные модели и масштабатор из файлов,
    расположенных в каталоге 'final_models'.
    """
    xgb_model = joblib.load('final_models/xgb_model.pkl')
    lgb_model = joblib.load('final_models/lgb_model.pkl')
    nn_model  = joblib.load('final_models/nn_model.pkl')
    scaler    = joblib.load('final_models/scaler.pkl')
    return xgb_model, lgb_model, nn_model, scaler

def load_client_data(file_path):
    """
    Загружает данные клиента из CSV-файла.
    Ожидается, что файл содержит одну строку с 64 числовыми значениями, разделёнными запятыми.
    Возвращает numpy-массив формы (1, 64).
    """
    try:
        # Если в файле нет заголовка, указываем header=None
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        sys.exit(1)

    if df.shape[1] != 64:
        print(f"Ошибка: ожидалось 64 признака, а получено {df.shape[1]}.")
        sys.exit(1)

    # Преобразуем DataFrame в numpy-массив размерности (1, 64)
    return df.values

def ensemble_predict(xgb_model, lgb_model, nn_model, scaler, client_features, threshold=0.5):
    """
    Принимает:
      - три обученные модели,
      - масштабатор,
      - numpy-массив с признаками клиента (1, 64),
      - пороговое значение (threshold) для принятия решения.

    Шаги:
      1. Масштабирует входной вектор признаков.
      2. Получает прогноз (вероятность дефолта, то есть класс 1) от каждой модели.
      3. Усредняет прогнозы.
      4. Если средняя вероятность >= threshold, кредит отклоняется,
         иначе – кредит одобряется.

    Выводит прогноз для каждой модели, среднюю вероятность и итоговое решение.
    """
    # Масштабирование входных данных
    client_features_scaled = scaler.transform(client_features)

    # Получаем вероятностные прогнозы для класса "дефолт" (индекс 1)
    proba_xgb = xgb_model.predict_proba(client_features_scaled)[0, 1]
    proba_lgb = lgb_model.predict_proba(client_features_scaled)[0, 1]
    proba_nn  = nn_model.predict_proba(client_features_scaled)[0, 1]

    # Выводим прогнозы каждой модели
    print("\nПрогнозы моделей:")
    print(f"XGBoost:       вероятность дефолта = {proba_xgb:.3f}")
    print(f"LightGBM:      вероятность дефолта = {proba_lgb:.3f}")
    print(f"NeuralNetwork: вероятность дефолта = {proba_nn:.3f}")

    # Усредняем вероятности
    avg_proba = (proba_xgb + proba_lgb + proba_nn) / 3.0
    print(f"\nСредняя вероятность дефолта: {avg_proba:.3f}")

    # Принимаем решение по пороговому значению (0.5 по умолчанию)
    decision = "Кредит отклонён" if avg_proba >= threshold else "Кредит одобрен"
    print(f"\nИтоговое решение: {decision}\n")

    return avg_proba, decision

def main():
    # Если путь к файлу не указан в аргументах, используем значение по умолчанию
    if len(sys.argv) < 2:
        file_path = "final_models/client_data.csv"  # Файл по умолчанию
        print(f"Путь к файлу не указан. Будет использован файл по умолчанию: {file_path}")
    else:
        file_path = sys.argv[1]

    # Шаг 1: Загрузка обученных моделей и масштабатора
    print("Загрузка обученных моделей...")
    xgb_model, lgb_model, nn_model, scaler = load_models()

    # Шаг 2: Загрузка данных клиента из файла
    print(f"Загрузка данных клиента из файла: {file_path}")
    client_features = load_client_data(file_path)

    # Шаг 3: Усреднение прогнозов моделей и принятие решения по кредиту
    ensemble_predict(xgb_model, lgb_model, nn_model, scaler, client_features, threshold=0.5)

if __name__ == "__main__":
    main()
