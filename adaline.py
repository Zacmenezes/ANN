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
        print("MSE=%f RMSE=%f" % (np.mean(errors), np.sqrt(np.mean(errors))))
        return np.mean(errors), np.sqrt(np.mean(errors)), test, predictions

    def evaluate(self, n):
        results = []
        for _ in range(n):
            results.append(self.realization())
        ms_errors = [r[0] for r in results]
        rms_errors = [r[1] for r in results]
        index = (np.abs(ms_errors)).argmin() 
        print("Melhor resultado: MSE=%f RMSE=%f" % (results[index][0], results[index][1]))     
        self.plot(results[index][2], results[index][3])

    def plot(self, data, predictions):
        if(self.problem.c == None):
            plt.scatter(data['x'].values, data['y'].values, c='blue', s=5)
            plt.scatter(data['x'].values, predictions, c='red', s=10)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(data['x'].values, data['y'].values, predictions, linewidth=0.2)
            ax.scatter(data['x'].values, data['y'].values, data['z'].values, c='red', marker='*')
        plt.show()

toy = Toy2(3, -4)
a = Adaline(toy, 0.1, 1000)
a.evaluate(5)