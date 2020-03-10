import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt



class LinearRegression():

    def __init__(self, learning_rate=0.001, epochs=10000):

        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0


    def fit(self, x, y):

        n_samples, n_features = x.shape


        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.epochs):

            y_prediction = np.dot(x, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(x.T, (y_prediction - y))
            db = (1 / n_samples) * np.sum(y_prediction - y)


            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, x):

        y_prediction = np.dot(x, self.weights) + self.bias
        
        return y_prediction


def mean_squared_error(gt, lb):

    return np.mean((gt - lb) ** 2)


if __name__ == '__main__':

    model = LinearRegression()

 
    x, y = datasets.make_regression(n_samples=100, n_features=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    error = mean_squared_error(y_test, predictions)

    print(f'Error Rate: {error}')