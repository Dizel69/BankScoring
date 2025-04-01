import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Импортируем модели и метрики из scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ЗАГРУЗКА ДАННЫХ
# Замените 'poland_processed.csv' на путь к вашему файлу
df = pd.read_csv('processed/poland_processed.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop(columns=['target'])
y = df['target']

# Разделение на обучающую и тестовую выборки (20% для теста)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Для моделей, чувствительных к масштабу, выполняем стандартизацию признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ
# Используем class_weight='balanced' для некоторых моделей, чтобы компенсировать дисбаланс классов.
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(),
    # 'CatBoost': CatBoostClassifier(verbose=0),  # Раскомментируйте, если установлен CatBoost
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

# Словарь для хранения ROC-кривых всех моделей
roc_curves = {}

# Список для хранения результатов (метрики)
results = []


# Функция для построения графиков для каждой модели
def plot_metrics(y_true, y_pred, y_proba, model_name):
    # Построение матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(f'{model_name} - Матрица ошибок')
    plt.xlabel('Предсказано')
    plt.ylabel('Фактически')
    plt.show()

    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {roc_auc:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Случайное предсказание')
    plt.xlabel('Ложноположительная доля (FPR)')
    plt.ylabel('Истинно положительная доля (TPR)')
    plt.title(f'{model_name} - ROC-кривая')
    plt.legend(loc='lower right')
    plt.show()

    # Построение Precision-Recall кривой
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label='Precision-Recall кривая', color='green')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title(f'{model_name} - Precision-Recall кривая')
    plt.legend(loc='lower left')
    plt.show()


# Обучение моделей и вывод графиков
for model_name, model in models.items():
    print(f"Обучаем модель: {model_name}...")
    # Используем масштабированные данные для моделей, чувствительных к масштабированию
    if model_name in ['LogisticRegression', 'SVM', 'NeuralNetwork', 'KNeighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Добавляем результаты в список
    results.append((model_name, acc, f1, auc))

    # Выводим отчёт по классификации
    print(f'=== {model_name} ===')
    print(f'Точность (Accuracy): {acc:.4f}, F1-score: {f1:.4f}, ROC-AUC: {auc:.4f}')
    print(classification_report(y_test, y_pred))
    print('-------------------------')

    # Сохраняем данные для ROC-кривой (будем использовать для общего графика)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[model_name] = (fpr, tpr, auc)

    # Выводим отдельные графики для модели
    plot_metrics(y_test, y_pred, y_proba, model_name)

# Построение сводной таблицы с результатами
df_results = pd.DataFrame(results, columns=['Модель', 'Точность', 'F1-score', 'ROC-AUC'])
print("Сводная таблица результатов:")
print(df_results)

# --- ОБЩИЙ ГРАФИК ROC-КРИВЫХ ДЛЯ ВСЕХ МОДЕЛЕЙ ---
plt.figure(figsize=(10, 8))
# Проходим по всем моделям и строим ROC-кривую каждой модели разным цветом
for model_name, (fpr, tpr, auc_value) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.3f})')
# Линия случайного предсказания
plt.plot([0, 1], [0, 1], 'k--', label='Случайное предсказание')
plt.xlabel('Ложноположительная доля (FPR)', fontsize=12)
plt.ylabel('Истинно положительная доля (TPR)', fontsize=12)
plt.title('Сравнение ROC-кривых для всех моделей', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.show()
