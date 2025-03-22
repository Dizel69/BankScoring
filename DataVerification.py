import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
df = pd.read_csv("dataset.csv")

# Выводим инфо по данным
print(df.info())

# Проверяем, есть ли пропущенные значения
print(df.isnull().sum())

# Смотрим распределение целевого класса
sns.countplot(x=df["target"])
plt.title("Распределение классов в датасете")
plt.show()

# Строим матрицу корреляции (первые 20 фич)
plt.figure(figsize=(12, 8))
sns.heatmap(df.iloc[:, :20].corr(), annot=False, cmap="coolwarm")
plt.title("Матрица корреляции первых 20 признаков")
plt.show()
