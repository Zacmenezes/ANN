import numpy as np
import pandas as pd
from iris_problem import Iris
from vertebral_problem import Vertebral
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from toy_problem import Toy
import math


class MLP():
    def __init__(self,
                 eta_initial=0.5,
                 eta_final=0.1,
                 n_output=3,
                 n_hidden=7,
                 max_epochs=200,
                 problem=None):
        self.eta_initial = eta_initial
        self.eta_final = eta_final
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

    def train_weights(self, train_df):
        train_data = train_df.drop(['d0', 'd1', 'd2'], axis=1).values
        target = train_df[['d0', 'd1', 'd2']].values

        W = np.random.uniform(low=-1.0, high=1.0, size=(self.n_hidden, train_data.shape[1]))
        M = np.random.uniform(low=-1.0, high=1.0, size=(self.n_output, self.n_hidden))

        for epoch in range(self.max_epochs):
            shuffle(train_data, target)
            learn_rate = self.eta(epoch)
            for row, row_target in zip(train_data, target):

                h = self.activation_function(np.dot(row, W.T))
                h_linha = self.derivate(h)

                y = self.activation_function(np.dot(h, M.T))
                y_linha = self.derivate(y)

                error = row_target - y

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
        W, M = self.train_weights(train)
        return train, test, W, M, self.hit_rate(test, W, M)

    def evaluate(self, n=20):
        for _ in range(n):
            r = self.realize()
            self.realizations.append(r)
            print(r[4])
        hit_rates = np.array(self.realizations)[:, 4].astype(float)
        acc = np.average(hit_rates)
        std = np.std(hit_rates, dtype=np.float32)
        index = (np.abs(hit_rates)).argmax()
        print("Accuracy=%f, Standard deviation=%f" % (acc, std))

    def validate(self, prediction):
        return [
            1 if output == np.amax(prediction) else 0
            for output in prediction
        ]

    def eta(self, epoch):
        return self.eta_initial * (self.eta_final/self.eta_initial)**(epoch/self.max_epochs)

# mlp = MLP(problem=Iris(),
#           n_output=3,
#           n_hidden=8,
#           max_epochs=400)

mlp = MLP(problem=Vertebral(),
          eta_initial=0.5,
          eta_final=0.1,
          n_output=3,
          n_hidden=12,
          max_epochs=500)

mlp.evaluate(n=10)