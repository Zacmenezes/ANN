import numpy as np
import pandas as pd
from iris_problem import Iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from toy_problem import Toy

class SingleLayerPerceptron():
 
    def __init__(self, learn_rate=0.1, neurons=3, max_epochs=200, problem=None):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.neurons = neurons
        self.problem = problem
        self.activation_function = np.vectorize(lambda x: 1 if x >= 0 else 0)

    def train_weights(self, train_df, learn_rate, max_epochs):
        train_data = train_df.drop(['d0','d1','d2'], axis=1).values
        target = train_df[['d0','d1','d2']].values
        weights = np.random.randn(self.neurons, train_data.shape[1])
        epoch = 0

        while(epoch < max_epochs):
            shuffle(train_data, target)
            for row, row_target in zip(train_data, target):  
                prediction = self.predict(row, weights.T)
                error = row_target - prediction
                weights = weights + learn_rate * np.outer(row, error).T
            epoch += 1
        return weights

    def hit_rate(self, data, trained_weights):
        actual = data[['d0','d1','d2']].values
        test = data.drop(['d0','d1','d2'], axis=1).values
        hit = 0
        for row, row_target in zip(test, actual):
            prediction = self.predict(row, trained_weights.T)
            if(prediction == row_target).all():
                hit += 1
        return (hit/ float(len(actual))) * 100.0

    def test(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        weights = self.train_weights(train, self.learn_rate, self.max_epochs)
        print(self.hit_rate(test, weights))
        self.plot_decision_surface(test, weights)
    
    def predict(self, row, weights):
        return self.validate(np.dot(row, weights))

    def validate(self, prediction):      
        if sum(self.activation_function(prediction)) != 1:
            return [1 if output == np.amax(prediction) else 0 for output in prediction]
        else:
            return self.activation_function(prediction)

    def plot_decision_surface(self, data, weights):
        c = data.columns
        actual = data[['d0','d1','d2']].values
        test = data.drop(['d0','d1','d2'], axis=1).values

        x1_max, x1_min = data[c[1]].max() + 0.2, data[c[1]].min() - 0.2
        x2_max, x2_min = data[c[2]].max() + 0.2, data[c[2]].min() - 0.2

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.09), np.arange(x2_min, x2_max, 0.09))
        Z =  np.array([xx1.ravel(), xx2.ravel()]).T
        
        fig, ax = plt.subplots()
        ax.set_facecolor((0.97, 0.97, 0.97))
        for x1, x2 in Z:
            if (self.predict([1, x1, x2], weights.T) == np.array([1, 0, 0])).all(): 
                ax.scatter(x1, x2, c='red', s=1.5, marker='o')
            elif (self.predict([1, x1, x2], weights.T) == np.array([0, 1, 0])).all():
                ax.scatter(x1, x2, c='green', s=1.5, marker='o')
            elif (self.predict([1, x1, x2], weights.T) == np.array([0, 0, 1])).all(): 
                ax.scatter(x1, x2, c='blue', s=1.5, marker='o')

        for row, row_target in zip(test, actual):
            if (row_target == np.array([1, 0, 0])).all():
                ax.scatter(row[1], row[2], c='red', marker='v')
            elif (row_target == np.array([0, 1, 0])).all():
                ax.scatter(row[1], row[2], c='green', marker='*')       
            elif (row_target == np.array([0, 0, 1])).all():
                ax.scatter(row[1], row[2], c='blue', marker='o')       
        plt.show()


problem = Iris(drop=['x1', 'x2'])
problem = Toy(neurons=3, class_size=50)

slp  = SingleLayerPerceptron(problem=problem, learn_rate=0.01, max_epochs=500)
slp.test()