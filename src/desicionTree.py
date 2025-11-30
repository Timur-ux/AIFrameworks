from src.Model import *
import numpy as np

DETALIZATION = 10
class DecisionTree(Model):
    class Node:
        def __init__(self, x: pd.DataFrame, y: pd.Series, currentDepth: int, maxDepth: int) -> None: 
            self.isLeaf = False
            self.column = None
            self.value = None
            self.left = None
            self.right = None
            currentEntropy = self.entropy(y)
            if currentDepth >= maxDepth or currentEntropy == 0:
                self.isLeaf = True
                uniques = y.unique()
                maxCount = -1
                for unique in uniques:
                    count = y[y == unique].shape[0]
                    if count > maxCount:
                        self.value = unique
                return

            # ищем наилучшее разбиение
            maxGain = -1e18
            maxCol = x.columns[0]
            maxVal = -1e9
            for column in x.columns:
                if column == "index":
                    continue
                min = x[column].min()
                max = x[column].max()
                delta = (max - min) / DETALIZATION
                for i in range(DETALIZATION):
                    currentVal = min + delta * i
                    left = y[x[x[column] < currentVal].index]
                    right = y[x[x[column] >= currentVal].index]
                    currentGain = currentEntropy - (left.shape[0] / y.shape[0] * self.entropy(left) + right.shape[0] / y.shape[0] * self.entropy(right))
                    if currentGain > maxGain:
                        maxGain = currentGain
                        maxCol = column
                        maxVal = currentVal
            self.column = maxCol
            self.value = maxVal


        @staticmethod
        def entropy(y: pd.Series):
            uniques = y.unique()
            result = 0
            for unique in uniques:
                p = y[y == unique].shape[0] / y.shape[0]
                result -= p * np.log2(p)

            return result



        def predict(self, x: pd.Series):
            if self.isLeaf:
                return self.value
            
            if x[self.column] < self.value:
                return self.left.predict(x)
            return self.right.predict(x)

    def __init__(self, maxDepth: int) -> None: 
        self.maxDepth: int = maxDepth

    @classmethod
    def split(cls, x: pd.DataFrame, y: pd.Series, currentDepth: int, maxDepth: int) -> Node:
        node = cls.Node(x, y, currentDepth, maxDepth)

        if currentDepth < maxDepth and not node.isLeaf:
            criteria = x[node.column] < node.value
            node.left = cls.split(x[criteria], y[criteria], currentDepth + 1, maxDepth)
            node.right = cls.split(x[~criteria], y[~criteria], currentDepth + 1, maxDepth)

        return node

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.root = self.split(x, y, 0, self.maxDepth)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        assert self.root is not None
        yPredicted = pd.Series(np.zeros(x.shape[0]))
        for i, (_, row) in enumerate(x.iterrows()):
            yPredicted[i] = self.root.predict(row)

        return yPredicted

