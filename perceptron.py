import numpy as np
import pandas as pd
from iris_problem import Iris

class Perceptron:

    def __init__(self, problem, l_rate, n_epoch):
        self.problem = problem
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def predict(self, row, weights):
        activation = np.dot(row, weights)
        return 1 if activation >= 0.0 else 0

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

    def split(self, data, frac):
        aux = data.sample(frac=frac)
        return aux, data.drop(aux.index)

    def hit_rate(self, data, trained_weights):
        hit = 0
        for row in data.values:
            actual = row[-1]
            prediction = self.predict(row[:-1], trained_weights)
            if actual == prediction:
                hit += 1
        return hit / float(len(data)) * 100.0

    def evaluate(self):
        hit_rates = []
        while(len(hit_rates) < 20):
            train, test = self.split(self.problem.data, 0.8)
            weights = self.train_weights(train, self.l_rate, self.n_epoch)
            hit_rates.append(self.hit_rate(test, weights))
        return np.average(hit_rates), np.std(hit_rates)

def main():
    
    setosa = Perceptron(Iris('Iris-setosa', 'data/iris.data'), 0.1, 20)
    versicolor = Perceptron(Iris('Iris-versicolor', 'data/iris.data'), 0.1, 20)
    virginica = Perceptron(Iris('Iris-virginica', 'data/iris.data'), 0.1, 20)

    print("Accuracy=%f, Standard deviation=%f" % setosa.evaluate())
    print("Accuracy=%f, Standard deviation=%f" % versicolor.evaluate())
    print("Accuracy=%f, Standard deviation=%f" % virginica.evaluate())
    
if __name__ == "__main__":
    main()