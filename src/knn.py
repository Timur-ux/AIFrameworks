import pandas as pd
import numpy as np

from src.Model import *
import numpy as np

class knn(Model):
    def __init__(self, kNeighbors: int) -> None: 
        assert kNeighbors > 0
        self.kNeighbors: int = kNeighbors

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.samplesFeatures = x
        self.samplesGroups = y

    def _predict(self, x: pd.Series) -> float:
        distances = np.sqrt(np.sum((self.samplesFeatures - x)**2, axis=1))
        kNearest = np.argsort(distances)[:self.kNeighbors]
        yPredicted = np.bincount(self.samplesGroups[kNearest]).argmax()
        return float(yPredicted)
    
    def predict(self, x: pd.DataFrame) -> pd.Series:
        if self.samplesFeatures is None:
            raise Exception("Predict must be called after model's fitting")
        result = []
        for _, xRow in x.iterrows():
            result.append(self._predict(xRow))
        return pd.Series(result)
    
    ##
    # @brief calc distance as sum of squares of difference
    @staticmethod
    def distance(row1: pd.Series, row2: pd.Series) -> float:
        assert row1.size == row2.size
        diff = row1 - row2
        return np.dot(diff, diff)
