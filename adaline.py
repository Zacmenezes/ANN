import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from toy_problem_adaline import Toy2
from mpl_toolkits.mplot3d import Axes3D as a3d

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
                weights = weights + l_rate * error * row[:-1]
            epoch += 1
        return weights

    def realization(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        weights = self.train_weights(train, self.l_rate, self.n_epoch)
        predictions = []
        errors = []
        for row in test.values:
            prediction = self.predict(row[:-1], weights)
            predictions.append(prediction)
            error = row[-1] - prediction
            errors.append(error * error)
        print("MSE=%f" % np.mean(errors))
        self.plot(test, predictions)

    def evaluate(self, n):
        for _ in range(n):
            self.realization()

    def plot(self, data, predictions):
        if(self.problem.c == None):
            for index, row in enumerate(data.values):
                plt.scatter(row[1], row[2], c='blue', s=10)
                plt.scatter(row[1], predictions[index], c='red', s=10)
            plt.show()
        else:
            z_surface = []
            for index, row in enumerate(data.values):
                z_surface.append(predictions[index])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(data['x'].values, data['y'].values, z_surface, linewidth=1, antialiased=True)
            ax.scatter(data['x'].values, data['y'].values, data['z'].values, c='red', marker='o')
            plt.show()

            

toy = Toy2(4, 8 ,-5)
a = Adaline(toy, 0.01, 1000)
a.realization()
# a.evaluate(20)