import numpy as np
import pandas as pd
from iris_problem import Iris
from sklearn.model_selection import train_test_split

class SingleLayerPerceptron():
 
    def __init__(self, learn_rate=0.1, neurons=3, max_epochs=200, problem=None):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.neurons = neurons
        self.problem = problem

    def train_weights(self, train_df, learn_rate, max_epochs):
        train_data = train_df.drop(['d0','d1','d2'], axis=1).values
        target = train_df[['d0','d1','d2']].values
        weights = np.random.randn(5,3)
        epoch = 0
        while(epoch < max_epochs):
            np.random.shuffle(train_data)
            for row, row_target in zip(train_data, target):   
                predictions = self.validate(np.dot(row, weights))
                error = row_target - predictions
                weights = weights + learn_rate * np.outer(row, error.T)
            epoch += 1
        return weights

    def hit_rate(self, data, trained_weights):
        actual = data[['d0','d1','d2']].values
        test = data.drop(['d0','d1','d2'], axis=1).values
        hit = 0
        for row, row_target in zip(test, actual):
            p = np.dot(row, trained_weights)
            p = self.validate(p)
            if((p == row_target).all()):
                hit += 1
        return (hit/ float(len(actual))) * 100.0

    def test(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        weights = self.train_weights(train, self.learn_rate, self.max_epochs)
        print(self.hit_rate(test, weights))
    
    def validate(self, prediction):
        degrau = np.vectorize(lambda x: 1 if x >= 0 else 0)
        if(sum(degrau(prediction)) != 1):
            return [1 if output == np.amax(prediction) else 0 for output in prediction]
        else:
            return degrau(prediction)


iris = Iris()
slp  = SingleLayerPerceptron(problem=iris, learn_rate=0.5, max_epochs=200)
slp.test()