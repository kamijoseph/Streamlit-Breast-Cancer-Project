
# Training the Logistic regression model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Model function
def build_model(data):
    # separating X feature and y labels
    X, y = data.drop(["diagnosis"], axis=1), data["diagnosis"]

    # scaling the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=21
    )

    # training the logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

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