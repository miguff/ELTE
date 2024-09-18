import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def main():

    life_exp = pd.read_csv("LifeExpectancyData.csv")
    print(life_exp.info())
    print(life_exp.describe())





if __name__ == "__main__":
    main()
