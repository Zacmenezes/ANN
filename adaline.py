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
        return np.mean(errors), np.sqrt(np.mean(errors)), test, predictions, weights

    def evaluate(self, n):
        results = []
        for _ in range(n):
            results.append(self.realization())
        ms_errors = [r[0] for r in results]
        rms_errors = [r[1] for r in results]
        index = (np.abs(ms_errors)).argmin() 
        print("MSE Standard deviation: %f RMSE Standard deviation: %f" % (np.std(ms_errors), np.std(rms_errors)))
        print("Melhor resultado: MSE=%f RMSE=%f" % (results[index][0], results[index][1]))     
        self.plot(results[index][2], results[index][4])

    def plot(self, data, weights):
       
        predictions = []

        if(self.problem.c == None):
            space = np.linspace(0, 1, 1000)
            for x in space:
                predictions.append(self.predict([1, x], weights))
            plt.scatter(data['x'].values, data['y'].values, c='blue', s=5)
            plt.scatter(space, predictions, c='red', s=5)
        else:
            space = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            Z =  np.array([space[0].ravel(), space[1].ravel()]).T
            for x, y in Z:
                predictions.append(self.predict([1, x, y], weights))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(space[0].ravel(), space[1].ravel(), predictions, linewidth=0.2, antialiased=True)
            ax.scatter(data['x'].values, data['y'].values, data['z'].values, c='red', marker='*')
            ax.set_xlabel('x', fontsize=20)
            ax.set_ylabel('y', fontsize=20)
            ax.set_zlabel('z', fontsize=20)
        plt.show()

toy = Toy2(5, -3, 20)
a = Adaline(toy, 0.01, 500)
a.evaluate(2)