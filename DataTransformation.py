import pandas as pd
import numpy as np
import os
import arff  # Требуется установить библиотеку liac-arff
from sklearn.preprocessing import LabelEncoder


##########################################
# ФУНКЦИИ ПРЕОБРАЗОВАНИЯ ДАТАСЕТОВ
##########################################

def process_german_numeric(file_path, numeric=True):
    """
    Обработка немецкого датасета.
    Если numeric=True, читается german.data-numeric (числовой вариант),
    иначе можно использовать оригинальный german.data с категориальными значениями.
    """
    sep = r'\s+' if numeric else ' '
    df = pd.read_csv(file_path, sep=sep, header=None)
    # Предположим, что в файле 25 колонок: 24 признака + 1 целевая
    num_features = df.shape[1] - 1
    new_columns = [f"f{i}" for i in range(1, num_features + 1)] + ["target"]
    df.columns = new_columns

    # Преобразование целевой переменной: 1 -> 0 (good), 2 -> 1 (bad)
    df["target"] = df["target"].apply(lambda x: 1 if x == 2 else 0)

    # Кодируем категориальные признаки (если есть)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])
    return df


def process_australia(file_path):
    """
    Обработка австралийского датасета (.dat).
    В файле 14 признаков + целевой атрибут. Разделитель - пробел или табуляция.
    Класс: '+' -> 1, '-' -> 0.
    """
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    new_columns = [f"f{i}" for i in range(1, 15)] + ["target"]
    df.columns = new_columns

    # Преобразование целевой переменной
    df["target"] = df["target"].apply(lambda x: 1 if str(x).strip() == '+' else 0)

    # Кодирование категориальных признаков
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])
    return df


def process_japan(file_path):
    """
    Обработка японского датасета.
    Часто встречаются варианты: crx.data, или файлы, где разделитель - запятая, либо пробел.
    Предполагается 15 признаков + целевой атрибут (итого 16 столбцов).
    Класс: '+' -> 1, '-' -> 0.
    """

    # 1) Сначала попробуем прочитать как CSV с запятой:
    try:
        df = pd.read_csv(file_path, sep=',', header=None)
        if df.shape[1] == 1:
            # Похоже, что данные "склеились" в одну колонку, значит запятая не подходит
            raise ValueError("CSV with comma gave 1 column, try whitespace.")
    except:
        # Если не получилось, пробуем заново с пробелами
        df = pd.read_csv(file_path, sep=r'\s+', header=None)

    # Теперь у нас должен быть >= 2 столбцов
    # Убедимся, что в итоге 16 столбцов: 15 фич + 1 таргет
    if df.shape[1] != 16:
        raise ValueError(
            f"Ожидается 16 столбцов (15 фич + 1 таргет), но получено {df.shape[1]}.\n"
            "Проверьте, что файл действительно японский датасет и что разделитель указан верно."
        )

    # Переименовываем столбцы
    new_columns = [f"f{i}" for i in range(1, 16)] + ["target"]
    df.columns = new_columns

    # Заменяем '?' на NaN (часто встречается)
    df.replace('?', np.nan, inplace=True)

    # Преобразуем целевую переменную: '+' -> 1, '-' -> 0
    df["target"] = df["target"].apply(lambda x: 1 if str(x).strip() == '+' else 0)

    # Кодируем категориальные признаки
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col].astype(str))

    # Заполним пропуски средними (если есть)
    for col in df.columns:
        if col != "target" and df[col].dtype in [float, int]:
            df[col].fillna(df[col].mean(), inplace=True)

    return df


def process_taiwan(file_path):
    """
    Обработка тайваньского датасета из Excel (.xls или .xlsx).
    Предполагается, что в файле имеется столбец ID, затем 23 признака и последний столбец — целевая переменная.
    """
    # Нужно установить 'xlrd' (для .xls) или 'openpyxl' (для .xlsx)
    df = pd.read_excel(file_path, header=1)
    # Удаляем столбец ID, если он есть
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    # Если общее число колонок = 24, то 23 признака + target
    num_features = df.shape[1] - 1
    new_columns = [f"f{i}" for i in range(1, num_features + 1)] + ["target"]
    df.columns = new_columns

    # Если классовые значения не числовые, можно их преобразовать:
    # Предположим, что значения уже 0/1.
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])
    return df


def process_poland(file_path):
    """
    Обработка польского датасета в формате ARFF.
    В файле описаны 64 числовых признака и целевой атрибут (class) {0,1}.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        arff_data = arff.load(f)

    data = arff_data['data']
    df = pd.DataFrame(data)

    num_features = df.shape[1] - 1
    new_columns = [f"f{i}" for i in range(1, num_features + 1)] + ["target"]
    df.columns = new_columns

    # Заменяем '?' на NaN и приводим к float
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)

    return df


##########################################
# ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ ОБРАБОТАННЫХ ДАННЫХ
##########################################

def save_dataset(df, name='dataset.csv', out_dir='processed'):
    """
    Сохраняет датафрейм в CSV-файл.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, name)
    df.to_csv(out_path, index=False)
    print(f"Сохранён файл: {out_path}")


##########################################
# ОСНОВНОЙ БЛОК ПРОГРАММЫ
##########################################

if __name__ == '__main__':
    # Пути к файлам (укажи корректные пути для своих файлов)
    path_german_numeric = 'data/german.data-numeric'
    path_australia = 'data/australian.dat'
    path_japan = 'data/japanese_credit.data'
    path_taiwan = 'data/taiwan.xls'
    path_poland = 'data/Polish_3year.arff'

    # Обработка каждого датасета
    try:
        df_german = process_german_numeric(path_german_numeric, numeric=True)
        save_dataset(df_german, 'german_processed.csv')
    except Exception as e:
        print("Ошибка при обработке немецкого датасета:", e)

    try:
        df_australia = process_australia(path_australia)
        save_dataset(df_australia, 'australia_processed.csv')
    except Exception as e:
        print("Ошибка при обработке австралийского датасета:", e)

    try:
        df_japan = process_japan(path_japan)
        save_dataset(df_japan, 'japan_processed.csv')
    except Exception as e:
        print("Ошибка при обработке японского датасета:", e)

    try:
        df_taiwan = process_taiwan(path_taiwan)
        save_dataset(df_taiwan, 'taiwan_processed.csv')
    except Exception as e:
        print("Ошибка при обработке тайваньского датасета:", e)

    try:
        df_poland = process_poland(path_poland)
        save_dataset(df_poland, 'poland_processed.csv')
    except Exception as e:
        print("Ошибка при обработке польского датасета:", e)

    print("Все датасеты обработаны и сохранены в директорию 'processed'.")
