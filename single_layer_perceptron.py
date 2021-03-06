import numpy as np
import pandas as pd
from iris_problem import Iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from toy_problem import Toy
import math

class SingleLayerPerceptron():
 
    def __init__(self, learn_rate=0.1, neurons=3, max_epochs=200, problem=None, activation='step'):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.neurons = neurons
        self.problem = problem
        self.realizations = []
        self.activation = activation
        if self.activation == 'step':
            self.activation_function = np.vectorize(lambda x: self.step(x))
            self.derivate = np.vectorize(lambda x: 1)    
        elif self.activation == 'logistic':
            self.activation_function = np.vectorize(lambda x: self.logistic(x))
            self.derivate = np.vectorize(lambda x: self.logistic(x) * (1 - self.logistic(x)))    
        elif self.activation == 'hiperbolic':
            self.activation_function = np.vectorize(lambda x: self.hiperbolic(x))
            self.derivate = np.vectorize(lambda x: 0.5 * (1 - self.hiperbolic(x) * self.hiperbolic(x)) )    

    def logistic(self, x):
        return 1 / (1 + math.exp(-x))

    def hiperbolic(self, x):
        return math.tanh(x)

    def step(self, x):
        return 1.0 if x >= 0.0 else 0.0    

    def train_weights(self, train_df, learn_rate, max_epochs):
        train_data = train_df.drop(['d0','d1','d2'], axis=1).values
        target = train_df[['d0','d1','d2']].values
        weights = np.random.randn(self.neurons, train_data.shape[1])

        for _ in range(max_epochs):
            shuffle(train_data, target)
            for row, row_target in zip(train_data, target):  
                prediction = self.predict(row, weights.T)
                error = row_target - prediction
                weights = weights + learn_rate * np.outer(row, error * self.derivate(prediction)).T        
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

    def realize(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        weights = self.train_weights(train, self.learn_rate, self.max_epochs)
        return train, test, weights, self.hit_rate(test, weights)
    
    def evaluate(self, n=20):
        for _ in range(n):
            self.realizations.append(self.realize())
        hit_rates = np.array(self.realizations)[:,3].astype(float)    
        acc = np.average(hit_rates)
        std = np.std(hit_rates, dtype=np.float32)
        index = (np.abs(hit_rates)).argmax()
        print("Activation=%s Accuracy=%f, Standard deviation=%f" % (self.activation, acc, std))
        # self.plot_decision_surface(self.realizations[index][1], self.realizations[index][2])

    def predict(self, row, weights):
        u = np.dot(row, weights)
        prediction = self.activation_function(u)
        aux = u if self.activation == 'step' else prediction
        return prediction if sum(prediction) == 1 else [1 if output == np.amax(aux) else self.problem.inhibit for output in aux]


    def plot_decision_surface(self, data, weights):
        c = data.columns
        actual = data[['d0','d1','d2']].values
        test = data.drop(['d0','d1','d2'], axis=1).values

        x1_max, x1_min = data[c[1]].max() + 0.2, data[c[1]].min() - 0.2
        x2_max, x2_min = data[c[2]].max() + 0.2, data[c[2]].min() - 0.2

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.04), np.arange(x2_min, x2_max, 0.04))
        Z =  np.array([xx1.ravel(), xx2.ravel()]).T
        
        fig, ax = plt.subplots()
        ax.set_facecolor((0.97, 0.97, 0.97))
        for x1, x2 in Z:
            prediction = self.predict([1, x1, x2], weights.T)
            if (prediction == np.array([1, self.problem.inhibit, self.problem.inhibit])).all(): 
                ax.scatter(x1, x2, c='red', s=1.5, marker='o')
            elif (prediction == np.array([self.problem.inhibit, 1, self.problem.inhibit])).all():
                ax.scatter(x1, x2, c='green', s=1.5, marker='o')
            elif (prediction == np.array([self.problem.inhibit, self.problem.inhibit, 1])).all(): 
                ax.scatter(x1, x2, c='blue', s=1.5, marker='o')

        for row, row_target in zip(test, actual):
            if (row_target == np.array([1, self.problem.inhibit, self.problem.inhibit])).all():
                ax.scatter(row[1], row[2], c='red', marker='v')
            elif (row_target == np.array([self.problem.inhibit, 1, self.problem.inhibit])).all():
                ax.scatter(row[1], row[2], c='green', marker='*')       
            elif (row_target == np.array([self.problem.inhibit, self.problem.inhibit, 1])).all():
                ax.scatter(row[1], row[2], c='blue', marker='o')       
        plt.show()



# slp  = SingleLayerPerceptron(problem=Iris(drop=['x1', 'x2']), learn_rate=0.01, max_epochs=500, activation='step')
# slp  = SingleLayerPerceptron(problem=Toy(neurons=3, class_size=50), learn_rate=0.01, max_epochs=500, activation='step')

# slp  = SingleLayerPerceptron(problem=Iris(drop=['x1', 'x2']), learn_rate=0.01, max_epochs=1000, activation='logistic')
# slp  = SingleLayerPerceptron(problem=Toy(neurons=3, class_size=50), learn_rate=0.01, max_epochs=500, activation='logistic')

# slp  = SingleLayerPerceptron(problem=Iris(drop=['x1', 'x2'], inhibit=-1), learn_rate=0.01, max_epochs=1000, activation='hiperbolic')
slp  = SingleLayerPerceptron(problem=Toy(neurons=3, class_size=50, inhibit=-1), learn_rate=0.05, max_epochs=600, activation='hiperbolic')


slp.evaluate(n=5)