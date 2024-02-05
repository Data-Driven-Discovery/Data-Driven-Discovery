
---
title: "Understanding the Basics of Artificial Intelligence"
date: 2024-02-05
tags: ['Artificial Intelligence', 'AI Algorithms', 'Beginner', 'Tutorial']
categories: ["basics"]
---


# Understanding the Basics of Artificial Intelligence

Hello reader, whether you're a data enthusiast, a machine learning developer, or someone looking to dip their toes into the exciting world of artificial intelligence(AI), you've come to the right place. In this article, we will provide a comprehensive overview of AI, including its origins, core concepts, and step-by-step instructions (with Python source code) on how to build a simplistic AI model.

## Introduction

Artificial Intelligence has been a fascinating concept of science fiction for decades; it has now become a day-to-day reality. Sophisticated technologies like self-driving cars, AI chatbots, recommendation systems, and voice assistants like Alexa and Siri – all are powerful manifestations of AI. AI allows devices to act smartly by learning from experiences, reshaping themselves and exhibiting human-like tasks.

[INSERT IMAGE HERE]

The image represents a typical data flow in an AI system, showing the integration of various components like data, machine learning, and deep learning, leading to AI.

## Brief Overview of AI

AI is a branch of computer science that aims to create systems capable of monitoring, learning, reasoning, problem-solving, perception, and language understanding. It involves programming computers to use characteristics of human intelligence. It divides mainly into two branches:

- Narrow AI: Systems designed to perform specific tasks such as voice recognition.
- General AI: Systems which can handle any intellectual task a human can do.

## AI and Machine Learning

Machine Learning (ML), an essential part of AI, is a method of data analysis that automates analytical model building. Leveraging algorithms that iteratively learn from data, ML allows systems to learn from experience. Here is a simple example of a Python script that employs the K-Nearest Neighbor (KNN) classifier from sklearn, a popular machine learning library.

```python
# Importing necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Loading breast cancer dataset
cancer = datasets.load_breast_cancer()

# Preprocessing the data
scaler = StandardScaler()
data = scaler.fit_transform(cancer.data)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    data, cancer.target, test_size=0.3, random_state=42)

# Using KNN classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)

# Displaying the score
print('Score: ', knn.score(X_test, Y_test))
```

By running the above script in Python, we'd see the model accuracy as output. This would be something similar to:

```
Score:  0.9649122807017544
```

This means our simplistic AI model achieved ~96% accuracy in making predictions – something unachievable without machine learning algorithms.

## Deep Learning: A Subset of Machine Learning

Deep Learning, a subset of ML, makes the computation of multi-layer neural networks feasible. It takes inspiration from the human brain's workings, creating patterns used for decision making. It's the key technology behind driverless cars, voice control systems, and image recognition. Here's a simple example of a deep learning model in Python using the TensorFlow library.

```python
# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compiling the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=5)

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
After executing this script, we'd get something like this:

```
Epoch 1/5
1875/1875 [==============================] - 2s 904us/step - loss: 0.2590 - accuracy: 0.9254
Epoch 2/5
1875/1875 [==============================] - 2s 892us/step - loss: 0.1143 - accuracy: 0.9660
Epoch 3/5
1875/1875 [==============================] - 2s 882us/step - loss: 0.0788 - accuracy: 0.9761
Epoch 4/5
1875/1875 [==============================] - 2s 881us/step - loss: 0.0588 - accuracy: 0.9819
Epoch 5/5
1875/1875 [==============================] - 2s 882us/step - loss: 0.0465 - accuracy: 0.9855
313/313 [==============================] - 0s 589us/step - loss: 0.0802 - accuracy: 0.9753
Test accuracy: 0.9753000140190125
```

As you see, our deep learning model achieved an accuracy of ~98% on the MNIST dataset - highly effective in recognizing handwritten digits.

## Conclusion

Artificial Intelligence, an exciting and transformative field, opens the doors to countless revolutionizing possibilities. While still in its infancy, AI promises brilliantly sophisticated solutions for problems globally. If you're a data professional willing to ride on this AI wave, understanding the basics of AI, machine learning, and deep learning is your first step. Who knows, the next big thing in AI could be your creation. Dive into the world of AI, machine learning, and deep learning. Happy AI journey!