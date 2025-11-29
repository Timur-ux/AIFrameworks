# %% [md]
"""
# ЛР 3 Дерево решений

Начнем с подгрузки датасетов и модели. Для загрузки датасетов я описал отдельные функциии. Они загружают датасеты также, как и в Preview.ipynb, только дополнительно сбрасывают индекс у получившихся DataFrame и Series

## Модель дерева решений

Моя реализация данной модели лежит в файле src.

"""

# %%
from src.desicionTree import *
# %%

from src.datasets import getClassificationDataset

defaultDataset, extendedDataset = getClassificationDataset()

# %% [md]
"""
Проверим точность на стандартном наборе данных
"""
# %%
from sklearn import metrics
model = DecisionTree(5)
model.fit(defaultDataset.xTrain, defaultDataset.yTrain)

yPredicted = model.predict(defaultDataset.xTest)
print(f"Accuracy: {metrics.accuracy_score(defaultDataset.yTest, yPredicted)}")
# %% [md]
"""
Проверим точность на расширенном наборе данных
"""
# %%
from sklearn import metrics
model = DecisionTree(5)
model.fit(extendedDataset.xTrain, extendedDataset.yTrain)

yPredicted = model.predict(extendedDataset.xTest)
print(f"Accuracy: {metrics.accuracy_score(extendedDataset.yTest, yPredicted)}")
# %% [md]
"""
# Вывод
Ну, дерево предсказывает. Проигрывает в точности реализации из sklearn примерно на 16 %. Расширение надора данных незначительно повышает точность
"""
