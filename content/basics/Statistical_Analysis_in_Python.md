
---
title: Statistical Analysis in Python: A Progressing Tool for Data Analysis
date: 2024-02-05
tags: ['Statistical Analysis', 'Python', 'Tutorial', 'Beginner']
categories: ["basics"]
---


# Statistical Analysis in Python: A Progressing Tool for Data Analysis

## Introduction

Statistical analysis is a fundamental component of data science. Besides it being the backbone of Machine Learning, its techniques allow us to get a clear picture of what data is trying to convey. Python, owing to its simplicity and extensive capabilities, has become a popular language for data analysis. It offers an array of libraries for performing statistical analysis, including Pandas, Numpy, Scipy, etc. 

In this article, we will explore how to use Python to perform statistical analysis. We will cover basics such as measures of centrality, dispersion while gradually moving to more complex aspects, like Statistical Hypothesis Tests. 

## Getting Started

Firstly, we will need to install necessary libraries. Pandas, Numpy, Scipy all come with the Anaconda Python Distribution. If you're not using Anaconda, execute the following commands to install them:

```bash
pip install pandas
pip install numpy
pip install scipy
```
Next, let's import the necessary libraries:

```python
import pandas as pd
import numpy as np
from scipy import stats
```

## Descriptive Statistics

We begin with descriptive statistics - a method to describe the basic features of the data and provide short summaries about the sample. Our focus will be on measures of central tendency, dispersion, and shape.

### Measures of Central Tendency

Let's create a pandas DataFrame for understanding.

```python
# Creating a pandas DataFrame.
data = pd.DataFrame({
    'A': np.random.normal(60, 10, 500),
    'B': np.random.normal(25, 5, 500),
    'C': np.random.normal(30, 7, 500)
})
data.head()
```

Output:

[INSERT IMAGE HERE]

'./images/data_head_out.png'

Let's calculate mean, median, and mode for each column.

```python
# Mean
mean = data.mean()
print("Mean:\n", mean) 

# Median
median = data.median()
print("\nMedian:\n", median)

# Mode
mode = data.mode()
print("\nMode:\n", mode)
```

The output showcases the average values for each column, providing insights into each dataset's behavior.

### Measures of Dispersion

We'll now focus on the standard deviation, variance, and the interquartile range (IQR) - defining the spread of the data.

```python
# Standard Deviation
std_dev = data.std()
print("Standard Deviation:\n", std_dev) 

# Variance
variance = data.var()
print("\nVariance:\n", variance)

# Interquartile Range
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print("\nIQR:\n", IQR)
```

Higher measures of dispersion would imply high variability in the data.

### Skewness and Kurtosis

These measure the shape of the probability distribution of a variable.

```python
# Skewness
skewness = data.skew()
print("\nSkewness:\n", skewness)

# Kurtosis
kurtosis = data.kurtosis()
print("\nKurtosis:\n", kurtosis)
```

## Statistical Hypothesis Testing

Statistical Hypothesis Tests are used to infer the results of a hypothesis performed on sample data from a larger population. We'll perform the Student's t-test.

### Student's t-test

```python
# Generating random Data:
np.random.seed(11)
data1 = 5 * np.random.randn(100) + 50
data2 = 5 * np.random.randn(100) + 51

# Perform t-test :
t_stat, p_val = stats.ttest_ind(data1, data2)
print("t-statistic :\n", t_stat)
print("\np-value :\n", p_val)
```

A p-value smaller than 0.05 means we can reject the null hypothesis.

## Conclusion

Python's simplicity, readability, and vast functionality make it an optimal language for performing statistical analysis in Data Science. With a basic understanding of Python, one can leverage the statistical libraries available to carry out extensive data analysis. With Python's enriched ecosystem and widespread community support, it will continue to be an asset for data professionals.

Remember the interpretation of these statistical measures hinges heavily on the context of the problem at hand. This article has covered the tip of the iceberg when it comes to the capabilities of Python in statistical analysis, and there's a lot more to explore and learn!