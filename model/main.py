
# Training the Logistic regression model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
        stratify=y,
        random_state=21
    )

    # training the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # testing the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    print(f"The Accuracy Score is:\n {accuracy}")
    print(f"\nClassification report:\n {clf_report}")

    return model, scaler


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
    # fetching and cleaning the data
    data = cleaned_data()

    # scaling and building the model
    model, scaler = build_model(data)


if __name__ == "__main__":
    main()