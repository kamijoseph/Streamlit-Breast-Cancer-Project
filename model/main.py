
# Training the Logistic regression model
import pandas as pd
import numpy as np

# data cleaning function
def cleaned_data():
    #importing the dataset
    data = pd.read_csv("dataset/breast_cancer.csv")

    # dropping unnamed: 32 column and the id column
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # encoding categorical variable: {diagnosis} into numerical values
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data

def main():
    data = cleaned_data()
    print(data.head())













if __name__ == "__main__":
    main()