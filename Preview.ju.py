# %% [md]
"""
# Описание датасетов:

1. Результаты теннисных матчей. Задача предсказания победителя. (Классификация)
2. Предсказание пола по весу и росту. (Регрессия)

## Метрики качества

В первом датасете будет оцениваться точность(accuracy) предсказания, т.е. доля правильных прогнозов, относительно их количества
"""

# %% [md]
"""
Рассмотрим сначала второй датасет с регрессией
"""

# %%

from sklearn import linear_model
from sklearn import model_selection, metrics
from sklearn import preprocessing
import pandas as pd
reg_df = pd.read_csv("./datasets/weight-height.csv")

print(f"Размер данных: {reg_df.shape}")
print(f"Пропуски в данных:\n{reg_df.isnull().sum()}")
print(f"Дубликаты в данных: {reg_df.duplicated().sum()}")

reg_df.info()
# %%
reg_df.sample()

# %% [md]
"""
Видим, что первая колонка является таргетом, остальные -- признаками. Выделим их в отдельные таблицы, заменим строковой таргет на целочисленный
"""

# %%

encoder = preprocessing.LabelEncoder()

x = reg_df[["Height", "Weight"]]
y = reg_df['Gender']

encoder.fit(y)
print(type(y), encoder.classes_)
y = pd.Series(encoder.transform(y))
type(y), y.head()

# %% [md]
"""
Разделим выборку на тестовую и обучающую части
"""

# %%
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
    x, y, test_size=0.3, random_state=51)

xTrain.shape, xTest.shape, yTrain.shape, yTest.shape

# %% [md]
"""
Обучим логистическую регрессию на наших данных
"""

# %%

model = linear_model.LogisticRegression(max_iter=1000)
model.fit(xTrain, yTrain)

# %% [md]
"""
Проверим точность предсказания на тестовой выборке
"""

# %%
yPredicted = model.predict(xTest)
accuracy = metrics.accuracy_score(yTest, yPredicted)
print(f"Accuracy: {accuracy:.2f}")

# %% [md]
"""
Получили бейзлайн в 92% правильных прогнозов

Теперь создадим бейзлайн для задачи классификации
"""

# %% 
class_df = pd.read_csv("./datasets/atp_tennis.csv")

print(f"Размер данных: {class_df.shape}")
print(f"Пропуски в данных:\n{class_df.isnull().sum()}")
print(f"Дубликаты в данных: {class_df.duplicated().sum()}")

class_df.info()
# %%
class_df.sample()

# %% [md]
"""
# Обзор данных

Рассмотрим распределение категорийных признаков(Tournament, Series, Court, Surface, Round, Score)
"""

# %%

import matplotlib.pyplot as plt
import seaborn as sns
categorical_cols = ['Tournament', 'Series', 'Court', 'Surface', 'Round', 'Score']
plt.figure(figsize=(20, 18))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    top_values = class_df[col].value_counts().nlargest(10)
    sns.barplot(y=top_values.index, x=top_values.values, palette='Set2', hue=top_values.index, legend=False)
    plt.title(f"Top 10 Values in '{col}'", fontsize=12)
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()

# %% [md]
"""
Рассмотрим расспределение числовых данных
"""
# %%
numerical_cols = ['Best of', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']
plt.figure(figsize=(20, 20))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 2, i)
    sns.histplot(class_df[col], kde=True, bins=30, color='#3498db')
    plt.title(f"Distribution of '{col}'", fontsize=12)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

# %% [md]
"""
## Создание бейзлайна по основным признакам

Для данных модели будем использовать три основных показателя(odds -- шансы, pts -- очки и rank -- рейтинг) у соперников

Будем предсказывать факт того, является будет ли первый игрок победителем
"""

# %%

class_df["Target"] = (class_df["Winner"] == class_df["Player_1"]).astype(int)

x = class_df[["Rank_1", "Rank_2", "Odd_1", "Odd_2", "Pts_1", "Pts_2"]]
y = class_df["Target"]

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.3, random_state=51)
xTrain.shape, xTest.shape, yTrain.shape, yTest.shape

# %% [md]
"""
Попробуем програть наши данные по всем классификаторам нашего курса
"""

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

names = ["knn", "decision tree", "random forest", "gradient boosting"]
models = [KNeighborsClassifier(), DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=51), RandomForestClassifier(max_depth=7, random_state=51), GradientBoostingClassifier(random_state=51)]

for name, model in zip(names, models):
    model.fit(xTrain, yTrain)
    yPredicted = model.predict(xTest)
    print(f"{name} baseline: {metrics.accuracy_score(yTest, yPredicted):.2f}")

# %% [md]
"""
# Улучшение бейзлайна

У меня есть гипотезы для задачи регрессии посчитать индекс массы тела, а для классификации -- отношения шансов, рейтинга и очков, а также их разницу(на счет разницы не уверен, т.к. это линейная комбинация признаков, но всё же)
"""

# %%
reg_df["IBM"] = reg_df['Weight'] / reg_df["Height"]
reg_df = reg_df.dropna()


class_df["Rank_diff"] = class_df['Rank_1'] - class_df['Rank_2']
class_df["Odd_diff"] = class_df['Odd_1'] - class_df['Odd_2']
class_df["Pts_diff"] = class_df['Pts_1'] - class_df['Pts_2']

class_df["Rank_ratio"] = class_df['Rank_1'] / class_df['Rank_2']
class_df["Odd_ratio"] = class_df['Odd_1'] / class_df['Odd_2']
class_df["Pts_ratio"] = class_df['Pts_1'] / class_df['Pts_2']
class_df = class_df.dropna()

# %% [md]
"""
Проверим новый бейзлайн
"""

# %%

# Регрессия
x = reg_df[["Height", "Weight", "IBM"]]
y = reg_df['Gender']

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
    x, y, test_size=0.3, random_state=51)

model = linear_model.LogisticRegression(max_iter=1000)
model.fit(xTrain, yTrain)

yPredicted = model.predict(xTest)
accuracy = metrics.accuracy_score(yTest, yPredicted)
print(f"Новая точность регрессии: {accuracy:.2f}")

# Классификация
x = class_df[["Rank_1", "Rank_2", "Odd_1", "Odd_2", "Pts_1", "Pts_2", "Rank_diff", "Odd_diff", "Pts_diff", "Rank_ratio", "Odd_ratio", "Pts_ratio"]]
y = class_df["Target"]

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.3, random_state=51)
names = ["knn", "decision tree", "random forest", "gradient boosting"]
models = [KNeighborsClassifier(), DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=51), RandomForestClassifier(max_depth=7, random_state=51), GradientBoostingClassifier(random_state=51)]

for name, model in zip(names, models):
    model.fit(xTrain, yTrain)
    yPredicted = model.predict(xTest)
    print(f"Модель {name}, новая точность: {metrics.accuracy_score(yTest, yPredicted):.2f}")

# %% [md]
"""
У модели решающего дерева по итогу точность повысилась на 0.01
"""
