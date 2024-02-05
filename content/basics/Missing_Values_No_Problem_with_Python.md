
---
title: "Missing Values? No Problem with Python"
date: 2024-02-05
tags: ['Data Cleaning', 'Python', 'Tutorial']
categories: ["basics"]
---


# Missing Values? No Problem with Python

Data is becoming the lifeblood of virtually all industries. In the era of big data, the quality of your dataset is crucial to the success of your predictive models. One of the frequent challenges faced by data scientists in preparing their data is dealing with missing values. Missing values are nothing but the absence of data values in a column. To be put simply, when there is no information, it's a missing value. In this article, we will navigate through handling missing values in python using various techniques and libraries including Pandas, NumPy, and scikit-learn. 

## Introduction
A dataset might contain missing values for a multitude of reasons including human errors during data collection, interruptions in the data flow, or sometimes certain features are only available for a portion of the population. Whatever the reason might be, missing values can lead to biased or incorrect results when not handled correctly. 

In python, there are various ways to check for and handle missing values. Let's start with a simple example:


```python
import pandas as pd
import numpy as np

# creating a simple dataframe with missing values
data = {'A':[1, 2, np.nan], 'B':[5, np.nan, np.nan], 'C':[1, 2, 3]}
df = pd.DataFrame(data)

print(df)
```
Output:
```
     A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
2  NaN  NaN  3
```

## Handling Missing Values

In the above output NaN, stands for Not a Number, and it is the missing value indicator in python. Now, let's see how we can handle these missing values.

### Drop missing values

The most simple solution to the missing values is to drop the rows or the entire column. **Pandas** library provides **dropna()** for that purpose.
```python
df.dropna()
```
Output:
```
     A    B  C
0  1.0  5.0  1
```
In the above case, **dropna()** drops all the rows with any column having null/no value.

### Filling the missing values

Another regular method of dealing with missing values is imputation - replacing missing values with substituted values. With the help of pandas **fillna()**, one can replace NaN values with the value they want.

```python
df.fillna(value='FILL VALUE')
```
Output:
```
           A          B  C
0          1          5  1
1          2  FILL VALUE  2
2  FILL VALUE  FILL VALUE  3
```

Also, we can fill missing values in a specified column with its mean value.
```python
df['A'].fillna(value=df['A'].mean())
```
Output:
```
0    1.0
1    2.0
2    1.5
Name: A, dtype: float64
```

### Using Scikit-learn 

Scikit-learn also provides various ways to impute missing values. `SimpleImputer` is a 2-step transformation i.e., first it fits to the data and then it transforms. It provides basic strategies for imputing missing values. 

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
SimpleImputer()

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))
```
Output:
```
array([[4. , 2. ],
       [6. , 3.66666667],
       [7. , 6. ]])
```

## Conclusion

Although dropping or filling missing values can be beneficial for preliminary data analysis, it is merely a simplistic approach and might not be the best option for handling missing values depending on the complexity of the dataset and the ML model. Further understanding of the underlying data is necessary in most cases. An understanding of the reasons behind the missing data may provide insight into how it can best be treated: for example, data may be missing due to equipment malfunction, in which case it may be appropriate to interpolate between data points, or perhaps to use the mean or median value. 

Remember, the ultimate goal is to create a predictive model that generalizes well to new unseen data. Therefore, it's not only about filling in missing values, but doing it in a manner that reflects the intricacies within your data.

In conclusion, the handling of missing values is one of the essential steps of the preprocessing data. Python offers an array of techniques to deal with missing data from Listwise or Pairwise deletion, to imputing with predictive models. The technique you choose should be primarily based on the nature of your data and the missingness.