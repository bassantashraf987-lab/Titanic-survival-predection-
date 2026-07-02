#import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#explore the dataset

sns.barplot(x="Sex", y="Survived", data=train)
plt.title("Survival Rate based on gender")
plt.show()
sns.barplot(x="Pclass", y="Survived", data=train)
plt.title("Survival Rate based on Passenger Class")
plt.show()
sns.histplot(train["Age"])
plt.title("Age Distribution of Passengers")
plt.show()

#data preprocessing
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
test["IsAlone"] = (test["FamilySize"] == 1).astype(int)
train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
train["Title"] = train["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major",
     "Rev","Sir","Jonkheer","Dona"],
    "Rare"
)

test["Title"] = test["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major",
     "Rev","Sir","Jonkheer","Dona"],
    "Rare"
)
train["Title"] = train["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major",
     "Rev","Sir","Jonkheer","Dona"],
    "Rare"
)

test["Title"] = test["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major",
     "Rev","Sir","Jonkheer","Dona"],
    "Rare"
)
train["Sex"] = train["Sex"].map({
    "male":0,
    "female":1
})

test["Sex"] = test["Sex"].map({
    "male":0,
    "female":1
})
train["Embarked"] = train["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})

test["Embarked"] = test["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})
title_mapping = {
    "Mr":1,
    "Miss":2,
    "Mrs":3,
    "Master":4,
    "Rare":5
}
#feature selection and model training
train["Title"] = train["Title"].map(title_mapping)
test["Title"] = test["Title"].map(title_mapping)
features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title"
]
X = train[features]

y = train["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
test_predictions = model.predict(test[features])
submission = test[["PassengerId"]].copy()

submission["Survived"] = test_predictions

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

#model evaluation and feature importance

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print(cm)

import pandas as pd

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print(importance)