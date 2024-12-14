# Machine Learning from scratch

This repository contains implementations of basic machine learning algorithms written from scratch, including:

- Linear Regression using Stochastic Gradient Descent (SGD) with Mean Absolute Error (MAE)
- Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE)
- Neural Network with Logistic Regression
- Titanic Dataset Problem using the above methods

The goal of this project is to understand the core principles behind these algorithms and demonstrate their implementation without relying on external libraries like `scikit-learn`.
## Datasets

The following datasets are included in this repository:

1. **TitanicTrainVal.csv**: Dataset for training and validating the Titanic classification model.
2. **classification_test.csv**: Test dataset for classification.
3. **classification_training.csv**: Training dataset for classification.
4. **window_heat.csv**: A dataset related to window heat (possibly for regression or classification).

## Algorithms Implemented

### 1. Linear Regression with SGD and MAE
This implementation solves the linear regression problem using stochastic gradient descent and optimizes the model using mean absolute error (MAE). This approach allows you to understand how gradient descent updates the parameters for minimizing the error.

### 2. SGD with MSE
An alternative implementation of linear regression using stochastic gradient descent but with mean squared error (MSE) as the loss function. MSE is widely used for regression tasks, and this implementation highlights how the choice of loss function can affect the optimization process.

### 3. Neural Network with Logistic Regression
A simple neural network is implemented from scratch to perform binary classification using logistic regression. This model uses a single hidden layer and is trained using backpropagation to minimize cross-entropy loss.

### 4. Titanic Problem
A Jupyter Notebook demonstrating the Titanic dataset problem. This notebook uses the aforementioned methods to predict passenger survival based on various features. It includes data preprocessing, feature selection, model training, and evaluation.

## Requirements

To run the code, you will need:

- Python 3.x
- NumPy
- Pandas
- Matplotlib (for visualization, if needed)
- Jupyter Notebook (for the Titanic problem)

You can install the necessary Python packages using pip:

```bash
pip install numpy pandas matplotlib jupyter


