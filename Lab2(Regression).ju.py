# %% [md]
"""
# ЛР 1 Regression

Начнем с подгрузки датасетов и модели.
Для загрузки датасетов я описал отдельные функциии. Они загружают датасеты также, как и в Preview.ipynb, только дополнительно сбрасывают индекс у получившихся DataFrame и Series

## Модель регрессии(логистической)

Моя реализация данной модели лежит в файле src.regression, я использовал стандартную вариант модели с сигмоидной функцией активации и стохастическим градиентным спуском в качестве алгоритма обучения
"""

# %%
from src.Model import *
import numpy as np
class Regression(Model):
    def __init__(self, nFeatures: int, lr: float = 1e-3, epochs: int = 100) -> None: 
        self.nFeatures: int = nFeatures
        self.lr: float = lr
        self.w = np.random.rand(nFeatures)
        self.bias = np.random.rand()

    @staticmethod
    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoidPrime(z: float) -> float:
        return Regression.sigmoid(z) * (1 - Regression.sigmoid(z))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        

