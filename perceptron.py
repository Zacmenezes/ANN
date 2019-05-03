import numpy as np
import pandas as pd
from iris_problem import Iris

class Perceptron:

    def __init__(self, problem):
        self.problem = problem

    def predict(self, row, weights):
        activation = np.dot(row, weights)
        return 1 if activation >= 0.0 else 0

    def train_weights(self, train, l_rate, n_epoch):
        train_values = train.values
        weights = np.append(-1 , np.ones(len(train.columns) - 2))
        for epoch in range(n_epoch):
            np.random.shuffle(train_values)
            for row in train_values:
                prediction = self.predict(row[:-1], weights)
                error = row[-1] - prediction
                weights = weights + l_rate * error * row[:-1]
        return weights

    def split(self, data, frac):
        aux = data.sample(frac=frac)
        return aux, data.drop(aux.index)

    def hit_rate(self, data, trained_weights):
        hit = 0
        for row in data:
            actual = row[-1]
            prediction = self.predict(row[:-1], trained_weights)
            if actual == prediction:
                hit += 1
        return hit / float(len(data)) * 100.0

    def accuracy(self):
        hit_rates = []
        while(len(hit_rates) < 20):
            train, test = self.split(self.problem.data, 0.8)
            weights = self.train_weights(train, 0.1, 20)
            hit_rates.append(self.hit_rate(test.values, weights))
        return np.average(hit_rates)

def main():
    setosa = Perceptron(Iris('Iris-setosa', 'data/iris.data'))
    versicolor = Perceptron(Iris('Iris-versicolor', 'data/iris.data'))
    virginica = Perceptron(Iris('Iris-virginica', 'data/iris.data'))
   
    print(setosa.accuracy())
    print(versicolor.accuracy())
    print(virginica.accuracy())
    
if __name__ == "__main__":
    main()