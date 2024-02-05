
---
title: "Classification Algorithms Every Data Scientist Should Know"
date: 2024-02-05
tags: ['Supervised Learning', 'Data Science', 'Machine Learning', 'Tutorial']
categories: ["basics"]
---


# Classification Algorithms Every Data Scientist Should Know

Classification algorithms are indeed the backbone of machine learning. They help in predicting the category or class of certain data points based on past observations. The reliability of their prediction is staggeringly good, making them widely used across several domains. This post will walk you through some pivotal classification algorithms every data scientist should know.

## Introduction

Before we begin, let's get a sense of what we are dealing with. By definition, classification is a two-step process consisting of learning step and prediction step. In the learning step, the model learns from the training dataset, and in the prediction step, it makes predictions on the input data. Some popular use cases of classification include email spam detection, bank customer churn prediction, tumor detection, etc.

Here, we will discuss five commonly used classification algorithms in the data science realm:

- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

We'll show small examples using `scikit-learn`, one of the most popular machine learning libraries in Python.

## 1. Logistic Regression

Despite its name, Logistic Regression (LR) is a fundamental classification method. It's simple and does not require high computational power as compared to complex models like SVM and Neural Networks.

Essentially, LR computes the weights of the variables that maximize the likelihood of predicting correct class probabilities when the dependent variable is binary.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```
The output score is a measure of accuracy on the test set. It lies between 0 and 1, with 1 indicating the best performance.

## 2. Decision Trees

Decision Trees are simple yet powerful algorithms used for both classification and regression problems. As the name suggests, a decision tree uses a tree-like model of decisions where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and the leaf nodes represent classes or class distributions.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
```

## 3. Random Forests

While Decision Trees are simple and easy to implement, they suffer from instability as small changes can lead to a completely different tree. This is where Random Forests step in. Random Forests mitigate this problem by combining the results of multiple decision trees built on different samples of the dataset.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```

## 4. Support Vector Machines (SVM)

SVM is a powerful, flexible but computationally expensive classification algorithm that can also handle continuous and categorical variables. SVM constructs hyperplanes in multidimensional space to separate different classes.

```python
from sklearn import svm

svc = svm.SVC()
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
```

## 5. K-Nearest Neighbors (KNN)

KNN is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It classifies new cases based on a similarity measure (e.g. distance functions).

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```

## Conclusion

The five classification algorithms discussed in this post are foundational knowledge for every data scientist. They come with their merits and demerits, and their selection depends primarily on the problem at hand. For more structured data, Decision Trees, Random Forests, and Logistic Regression can be good starting points. For less structured data, more complex models like SVM or Neural Networks can be considered. As always in machine learning and data science, the golden rule is: there's no free lunch. Each algorithm shines in certain domains and conditions, so it requires experience and continuous learning to select the right algorithm for specific tasks.