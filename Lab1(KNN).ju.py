# %% [md]
"""
# ЛР 1 KNN

Начнем с подгрузки датасетов и модели.
Для загрузки датасетов я описал отдельные функциии. Они загружают датасеты также, как и в Preview.ipynb, только дополнительно сбрасывают индекс у получившихся DataFrame и Series

## Модель knn

Моя реализация данной модели лежит в файле src.knn, я использовал идею brute-force -- простого перебора с евклидовой метрикой

"""
# %%
from src.datasets import getClassificationDataset
from src.knn import *
from sklearn import metrics

defaultDataset, extendedDataset = getClassificationDataset()
defaultDataset.xTrain.index.max(), defaultDataset.yTest.index.max()
model = knn(5)
model.fit(defaultDataset.xTrain, defaultDataset.yTrain)

# %% [md]
"""
Посчитаем точность предсказания на стандартном наборе данных
"""

# %%
yPredicted = model.predict(defaultDataset.xTest)
print(f"knn accuracy: {metrics.accuracy_score(defaultDataset.yTest, yPredicted):.2f}")

# %% [md]
"""
Проверим точность на расширенном наборе данных
"""
# %%
model = knn(5)
model.fit(extendedDataset.xTrain, extendedDataset.yTrain)
yPredicted = model.predict(extendedDataset.xTest)
print(f"knn accuracy: {metrics.accuracy_score(extendedDataset.yTest, yPredicted):.2f}")

# %% [md]
"""
# Вывод
Точность таже, время работы больше примерно на 2 порядка
"""
