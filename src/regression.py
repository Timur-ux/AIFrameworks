EPS = 1e-5
from src.Model import *
import numpy as np

class Regression(Model):
    def __init__(self, nFeatures: int, lr: float = 1e-3, epochs: int = 100, lmbda = 1e-2) -> None: 
        self.lmbda: float = lmbda
        self.nFeatures: int = nFeatures
        self.lr: float = lr
        self.w = np.random.rand(nFeatures)
        self.bias = np.random.rand()
        self.epochs = epochs

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoidPrime(x):
        return Regression.sigmoid(x) * (1 - Regression.sigmoid(x))

    @staticmethod
    def normalize(arr):
        return (arr - arr.mean(axis=0)) / np.sqrt(arr.var(axis=0) + EPS)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.costs = []
        minCost = 1e9
        minW, minBias = self.w, self.bias
        ones = np.ones((x.shape[0], 1), dtype=np.float32)
        x_, y_ = np.concat([x.to_numpy(dtype=np.float32), ones], axis=1), y.to_numpy()
        x_, y_ = Regression.normalize(x_), Regression.normalize(y_)
        for epoch in range(self.epochs):
            wb = np.concat([self.w, [self.bias]])
            # Проход вперед
            a = x_ @ wb
            z = Regression.sigmoid(a)

            # Проход назад
            zPrime = Regression.sigmoidPrime(a)

            primeFactor = np.sum(2 * (z - y_) * zPrime, axis=0) / z.shape[0] 
            
            biasPrime = primeFactor + self.lmbda * 2*self.bias
            weightsPrime = np.sum(primeFactor * x_, axis=0)[:-1] + self.lmbda * 2*self.w
            
            self.bias -= self.lr * biasPrime
            self.w    -= self.lr * weightsPrime

            self.costs.append(Regression.cost(y_, self.predict(x).to_numpy()))
            if self.costs[-1] < minCost:
                minCost = self.costs[-1]
                minW = self.w
                minBias = self.bias

        print(f"Min cost = {minCost} reached at {self.costs.index(minCost)} iteration restored as best model state")
        self.w = minW
        self.bias = minBias
                

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(Regression.sigmoid(Regression.normalize(x.to_numpy()) @ self.w.transpose() + self.bias).round())

    @staticmethod
    def cost(y: np.ndarray, yPred: np.ndarray):
        return np.sum((y - yPred) ** 2, axis=0) / y.shape[0]

