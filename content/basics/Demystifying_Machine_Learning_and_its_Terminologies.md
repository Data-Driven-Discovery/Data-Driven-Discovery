
---
title: Demystifying Machine Learning and its Terminologies
date: 2024-02-05
tags: ['Machine Learning', 'Data Science', 'Tutorial', 'Beginner']
categories: ["basics"]
---


# Demystifying Machine Learning and its Terminologies

In today's data-driven world, Machine Learning (ML) has become a buzzword, but do we really understand what it means, and its potential? Are we aware of the various terminologies used in this field, which play a crucial role in understanding and implementing Machine Learning concepts? This article focuses on simplifying and explaining Machine Learning and its related terminologies.

## Introduction

Machine Learning (ML) is a subset of artificial intelligence (AI) that gives computers the ability to learn from data without being explicitly programmed. With Machine Learning, we can achieve complex tasks like speech recognition, image recognition, recommendation systems, which were considered difficult a few years back.

The terminologies used in ML are essential to comprehend how various models and algorithms function, and their subsequent utilization in finding solutions to complex problems.

Let's walk through some of the most common terminologies used in Machine Learning.

## Machine Learning Terminologies

### 1. Dataset

A dataset is a collection of data which is usually formatted in a table. The dataset is the primary resource used to train and evaluate machine learning models. 

In Python, we can generate a simple dataset using the `pandas` library.

```python
import pandas as pd 

data = {'Name':['Tom', 'Nick', 'John', 'Tom'], 
        'Age':[20, 21, 19, 20]} 

df = pd.DataFrame(data)

print(df)
```
Output
```
   Name  Age
0   Tom   20
1  Nick   21
2  John   19
3   Tom   20
```

### 2. Features

Features are individual properties or characteristics of a phenomenon which are used as input in machine learning models. 

In the above dataset, 'Age' is a feature.

### 3. Training

Training a model simply means learning (estimating) the parameters of the model on the training dataset. For instance, let's create a simple linear regression model using `scikit-learn`.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

#We define X as the feature
X = np.array(df['Age']).reshape((-1, 1))

#We define y as the target we wish to predict
y = np.array(df['Name'])

model = LinearRegression().fit(X, y)
```

### 4. Evaluation

Evaluation involves assessing the performance of the trained model using a test dataset. Let's assume we have our test dataset `X_test`, the prediction would be ;

```python
y_pred = model.predict(X_test)
```

### 5. Overfitting & Underfitting

Overfitting refers to a model that models the training data too well, capturing the noise along with the underlying pattern in data. Underfitting, on the other hand, refers to when our model is too simple to capture underlying trends in the data.

### 6. Regularization

Regularization is a technique used to prevent overfitting by adding a complexity term to the loss function. 


### 7. Supervised Learning & Unsupervised Learning

Supervised learning is where we have input variables (x) and an output variable (Y) and we use an algorithm to learn the mapping function from the input to the output: Y = f(X).

Unsupervised learning is where we only have input data (X) and no corresponding output variables. This can be seen as learning the inherent structure of our data.

## Conclusion 

While the landscape of Machine Learning may seem overwhelming, it is truly a fascinating field full of potential and possibilities. As we continue to strive for technological advancements, a solid grasp of Machine Learning and its terminologies becomes extremely crucial. Keep learning and keep exploring this field to decipher its capabilities!

_[INSERT IMAGE HERE](./image.png)_