from typing_extensions import List
from src.desicionTree import DecisionTree
from src.Model import *
import numpy as np
import random
from statistics import multimode

class RandomForest(Model):
    def __init__(self, nTrees: int, sampleRatio: float, featureRatio: float, treeMaxDepth: int) -> None: 
        self.nTrees: int = nTrees
        self.sampleRatio: float = sampleRatio
        self.featureRatio: float = featureRatio
        self.treeMaxDepth: int = treeMaxDepth
        
        self.trees: List[DecisionTree] = []
        for i in range(nTrees):
            self.trees.append(DecisionTree(treeMaxDepth))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        features = list(filter(lambda x: x != "index", x.columns))
        nFeatureToSample = max(1, min(int(len(features) * self.featureRatio), len(features)))
        nSamples = max(1, min(x.shape[0], int(x.shape[0]*self.sampleRatio)))

        for i in range(self.nTrees):
            currentFeatures = random.sample(features, nFeatureToSample)
            x_ = x[currentFeatures].sample(nSamples)
            y_ = y[x_.index]
            self.trees[i].fit(x_, y_)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        results = np.zeros((self.nTrees, x.shape[0]))
        for i, tree in enumerate(self.trees):
            result = tree.predict(x).to_numpy()
            results[i] = result


        frequencies = list(map(multimode, results.T))
        result = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            result[i] = 1 - frequencies[i][0]

        return pd.Series(result)


