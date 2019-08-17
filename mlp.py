import numpy as np
import pandas as pd
from iris_problem import Iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from toy_problem import Toy
import math


class MLP():
    def __init__(self,
                 learn_rate=0.1,
                 n_output=3,
                 n_hidden=7,
                 max_epochs=200,
                 problem=None):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.problem = problem
        self.realizations = []
        self.activation_function = np.vectorize(lambda x: self.logistic(x))
        self.derivate = np.vectorize(lambda x: self.logistic(x) *
                                     (1 - self.logistic(x)))

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def train_weights(self, train_df, learn_rate, max_epochs):
        train_data = train_df.drop(['d0', 'd1', 'd2'], axis=1).values
        target = train_df[['d0', 'd1', 'd2']].values

        W = np.random.randn(self.n_hidden, train_data.shape[1])
        M = np.random.randn(self.n_output, self.n_hidden)

        for _ in range(max_epochs):
            shuffle(train_data, target)
            for row, row_target in zip(train_data, target):

                h = self.activation_function(np.dot(row, W.T))
                u = self.activation_function(np.dot(h, M.T))

                y = self.validate(u)
                error = row_target - y

                h_linha = self.derivate(h)
                y_linha = self.derivate(y)

                y_linha_times_error = y_linha * error
                herror = np.dot(M.T, y_linha_times_error)

                M = M + learn_rate * (np.outer(h, error) * y_linha).T
                W = W + learn_rate * (np.outer(row, herror) * h_linha).T

        return W, M

    def hit_rate(self, data, W, M):
        actual = data[['d0', 'd1', 'd2']].values
        test = data.drop(['d0', 'd1', 'd2'], axis=1).values
        hit = 0
        for row, row_target in zip(test, actual):
            h = self.activation_function(np.dot(row, W.T))
            u = self.activation_function(np.dot(h, M.T))

            y = self.validate(u)
            if (y == row_target).all():
                hit += 1
        return (hit / float(len(actual))) * 100.0

    def realize(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        W, M = self.train_weights(train, self.learn_rate, self.max_epochs)
        return train, test, W, M, self.hit_rate(test, W, M)

    def evaluate(self, n=20):
        for _ in range(n):
            self.realizations.append(self.realize())
        hit_rates = np.array(self.realizations)[:, 4].astype(float)
        acc = np.average(hit_rates)
        std = np.std(hit_rates, dtype=np.float32)
        index = (np.abs(hit_rates)).argmax()
        print("Accuracy=%f, Standard deviation=%f" % (acc, std))

    def validate(self, prediction):
        return [
            1 if output == np.amax(prediction) else self.problem.inhibit
            for output in prediction
        ]


mlp = MLP(problem=Iris(),
          learn_rate=0.1,
          n_output=3,
          n_hidden=12,
          max_epochs=600)

mlp.evaluate(n=10)