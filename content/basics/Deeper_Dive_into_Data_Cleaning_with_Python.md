
---
title: Deeper Dive into Data Cleaning with Python
date: 2024-02-05
tags: ['Data Cleaning', 'Python', 'Data Science', 'Tutorial']
categories: ["basics"]
---


# Deeper Dive into Data Cleaning with Python

Harnessing the power of machine learning and data science often begins with one critical step: data cleaning. Raw data is often messy and packed with inefficiencies and inaccuracies, not to mention irrelevant information that can skew your models and findings if left unchecked. Successful data professionals always make time for thorough data cleaning as the data foundation determines the quality and precision of end products. Today, we take a deeper dive into data cleaning with Python, one of the most potent languages leveraged in machine learning.

## Introduction 

Data cleaning, also known as data cleansing, involves identifying and rectifying errors and discrepancies from data to improve its quality. The mistakes could range from incorrect data like birthdays landing in the future to inconsistencies in the naming conventions. 

Python, a popular language for data science, affords an array of libraries to ease data cleaning efforts. This article focuses on three: pandas, NumPy, and scikit-learn. 

Before we dive in, it's important to note that data cleaning is project-specific—what applies in one situation might not work in another. This tutorial serves as a guide and points out some common techniques and libraries to streamline your data cleaning process.

## Exploring and Cleaning our Dataset

Most of the examples in this article will utilize a small example of a dataset that could possibly contain missing values, duplicates, incorrect data, or other inconsistencies. However, for privacy purposes the dataset originated from made up values. 

First, let's import the necessary libraries and the dataset.

```python
import pandas as pd
import numpy as np

# Let's create a small sample dataframe
data = {'Name': ['John', 'Anna', 'Peter', 'John', 'Anna'],
        'Age': [34, np.nan, 56, 34, 28],
        'City': ['New York', 'Los Angeles', np.nan, 'New York', '']}
df = pd.DataFrame(data)
print(df)
```

Output:

```python
    Name   Age         City
0   John  34.0     New York
1   Anna   NaN  Los Angeles
2  Peter  56.0          NaN
3   John  34.0     New York
4   Anna  28.0             
```
You can see that our toy dataset has some duplicated entries, missing data, and a blank entry. Let's tackle these.

### Handling Duplicates

It's important to deal with duplicates to prevent giving some information more weight than is warranted. To find and remove duplicates in pandas, `duplicated()` and `drop_duplicates()` methods come in handy.

```python
# Detect duplicate rows
duplicate_rows = df.duplicated()
print(duplicate_rows)
```

Output:

```python
0    False
1    False
2    False
3    True
4    False
dtype: bool
```

To drop the duplicates:

```python
# Drop duplicate rows
df.drop_duplicates(inplace=True)
print(df)
```

Output:

```python
    Name   Age         City
0   John  34.0     New York
1   Anna   NaN  Los Angeles
2  Peter  56.0          NaN
4   Anna  28.0             
```

Here we have successfully eliminated the duplicate row.

### Handling Missing Data

Missing data is a reality of the data analysis process. Techniques to deal with them include deleting rows or columns with missing data, filling in the missing values, or leaving them as is, depending on the analysis.

```python
# Detect missing values
print(df.isnull().sum())
```

Output:

```python
Name    0
Age     1
City    1
dtype: int64
```
Choosing to fill missing values:

```python
# Fill missing values with a specified value
df_filled = df.fillna({"Age": df["Age"].mean(), "City": "Not specified"})
print(df_filled)
```

Output:

```python
    Name   Age           City
0   John  34.0       New York
1   Anna  39.333  Los Angeles
2  Peter  56.0   Not specified
4   Anna  28.0               
```

We've filled missing age with the mean age and missing cities with "Not specified."

### Handling Inconsistencies and More

Some elements like the blank entry under 'City' for Anna, while not identified as missing (NaN), still constitute improper formatting.

```python
# Replace blank entries with 'Not specified'
df_filled.replace("", "Not specified", inplace=True)
print(df_filled)
```

Output:

```python
    Name   Age           City
0   John  34.0       New York
1   Anna  39.333  Los Angeles
2  Peter  56.0   Not specified
4   Anna  28.0   Not specified
```

## Conclusion

Data cleaning can make or break your machine learning algorithms and models, making it an indispensable step in data-driven projects. Despite the simplicity of our dataset, we've shown some of how Python's pandas library can streamline detection and resolution of common issues in raw data, from duplicates to missing data and format inconsistencies.

Remember, though, that data cleaning is project- and dataset-specific, and there's more to it than we've covered here. Outliers, irrelevant data, correct data representation, and more, should still be top of mind. Nonetheless, these examples should give you a strong starting point, to dive deeper into data cleaning with Python. Happy Cleaning!