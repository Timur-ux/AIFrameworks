from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        ...

    
