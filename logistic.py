from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():

    def __init__(self, learning_rate=0.001, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, x, y):

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)

        self.bias = 0


        for i in range(self.epochs):

            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, x):

        linear_approx = np.dot(x, self.weights) + self.bias
        prediction = self.sigmoid(linear_approx)

        real_approx = list(map(lambda x: 1 if x > 0.5 else 0, prediction))

        return real_approx

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    model = LogisticRegression()

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 1234)
 
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    
    accuracy = np.sum(y_test == predictions) / len(y_test)
    error_rate = np.sum(y_test != predictions) / len(y_test)

    print(f'Accuracy: {accuracy}')
    print(f'MSError : {error_rate}')
    
   






