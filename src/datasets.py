import pandas as pd
from sklearn import preprocessing, model_selection
from pandas.util.version import Tuple

EPS = 1e-5

class Dataset:
    def __init__(self, xTrain: pd.DataFrame, xTest: pd.DataFrame, yTrain: pd.Series, yTest: pd.Series) -> None: 
        self.xTrain: pd.DataFrame = xTrain
        self.xTest: pd.DataFrame = xTest
        self.yTrain: pd.Series = yTrain
        self.yTest: pd.Series = yTest

##
# @brief Loads regression dataset and splittes it by given ratio
#
# @param testSize size of test part
#
# @return tuple of source dataset and dataset with new features
def getRegressionDataset(testSize: float = 0.3) -> Tuple[Dataset, Dataset]:
    reg_df = pd.read_csv("./datasets/weight-height.csv")
    x = reg_df[["Height", "Weight"]]
    y = reg_df['Gender']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    y = pd.Series(encoder.transform(y))
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
        x, y, test_size=testSize, random_state=51)
    defaultDataset = Dataset(xTrain, xTest, yTrain, yTest)

    reg_df["IBM"] = reg_df['Weight'] / (reg_df["Height"] + EPS)
    reg_df = reg_df.dropna()
    x = reg_df[["Height", "Weight", "IBM"]]
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
        x, y, test_size=testSize, random_state=51)
    extendedDataset = Dataset(xTrain, xTest, yTrain, yTest)

    return defaultDataset, extendedDataset

def split(x:pd.DataFrame, y: pd.Series, testSize: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=testSize, random_state=51)
    return xTrain, xTest, yTrain, yTest


##
# @brief Loads classification dataset and splittes it by given ratio
#
# @param testSize size of test part
#
# @return tuple of source dataset and dataset with new features
def getClassificationDataset(testSize: float = 0.3) -> Tuple[Dataset, Dataset]:
    class_df = pd.read_csv("./datasets/atp_tennis.csv")
    class_df["Target"] = (class_df["Winner"] == class_df["Player_1"]).astype(int)

    x = class_df[["Rank_1", "Rank_2", "Odd_1", "Odd_2", "Pts_1", "Pts_2"]]
    y = class_df["Target"]

    xTrain, xTest, yTrain, yTest = split(x, y, testSize)
    xTrain.reset_index(inplace=True); xTest.reset_index(inplace=True); yTrain = yTrain.reset_index(drop=True); yTest = yTest.reset_index(drop=True)
    defaultDataset = Dataset(xTrain, xTest, yTrain, yTest)

    class_df["Rank_diff"] = class_df['Rank_1'] - class_df['Rank_2']
    class_df["Odd_diff"] = class_df['Odd_1'] - class_df['Odd_2']
    class_df["Pts_diff"] = class_df['Pts_1'] - class_df['Pts_2']

    class_df["Rank_ratio"] = class_df['Rank_1'] / (class_df['Rank_2'] + EPS)
    class_df["Odd_ratio"] = class_df['Odd_1'] / (class_df['Odd_2'] + EPS)
    class_df["Pts_ratio"] = class_df['Pts_1'] / (class_df['Pts_2'] + EPS)
    class_df = class_df.dropna()

    x = class_df[["Rank_1", "Rank_2", "Odd_1", "Odd_2", "Pts_1", "Pts_2", "Rank_diff", "Odd_diff", "Pts_diff", "Rank_ratio", "Odd_ratio", "Pts_ratio"]]
    y = class_df["Target"]

    xTrain, xTest, yTrain, yTest = split(x, y, testSize)
    xTrain.reset_index(inplace=True); xTest.reset_index(inplace=True); yTrain = yTrain.reset_index(drop=True); yTest = yTest.reset_index(drop=True)
    extendedDataset = Dataset(xTrain, xTest, yTrain, yTest)
    return defaultDataset, extendedDataset



