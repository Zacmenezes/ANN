import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from toy_problem_adaline import Toy2

class Adaline:
    
    def __init__(self, problem, l_rate, n_epoch):
        self.problem = problem
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.weights = []
        self.test = []

    def predict(self, row, weights):
        return np.dot(row, weights)

    def train_weights(self, train_df, l_rate, n_epoch):
        train_values = train_df.values
        weights = np.random.randn(len(train_df.columns) - 1)
        epoch = 0
        while(epoch < n_epoch):
            np.random.shuffle(train_values)
            for row in train_values:
                prediction = self.predict(row[:-1], weights)
                error = row[-1] - prediction
                print(weights)
                weights = weights + l_rate * error * row[:-1]
            epoch += 1
        return weights

    def realization(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        weights = self.train_weights(train, self.l_rate, self.n_epoch)
        predicted = []
        for row in test.values:
            p = self.predict(row[:-1], weights)
            plt.scatter(row[1], row[2], c='blue')
            plt.scatter(row[1], p, c='red')
        plt.show()

toy = Toy2(5,-3)
a = Adaline(toy, 0.001, 100)
a.realization()
