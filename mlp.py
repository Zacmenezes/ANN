import numpy as np
import pandas as pd
from iris_problem import Iris
from vertebral_problem import Vertebral
from dermatology_problem import Dermatology
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from toy_problem import Toy
import math


class MLP():
    def __init__(self,
                 n_hidden=[8],
                 eta_initial=0.5,
                 eta_final=0.1,
                 max_epochs=200,
                 problem=None):
        self.n_hidden = n_hidden
        self.eta_initial = eta_initial
        self.eta_final = eta_final
        self.max_epochs = max_epochs
        self.problem = problem
        self.realizations = []
        self.activation_function = np.vectorize(lambda x: self.logistic(x))
        self.derivate = np.vectorize(lambda x: self.logistic(x) *
                                     (1 - self.logistic(x)))

    def initLayers(self):
        layers = []
        input_size = len(self.problem.data_labels)

        # Hidden layers
        for layer_index in range(len(self.n_hidden)):
            W = np.random.uniform(low=-1.0,
                                  high=1.0,
                                  size=(input_size,
                                        self.n_hidden[layer_index]))
            layers.append({
                'weights': W,
                'h': None,
                'h_derivate': None,
                'error': None
            })
            input_size = W.shape[1]

        # Output layer
        M = np.random.uniform(low=-1.0,
                              high=1.0,
                              size=(self.n_hidden[len(self.n_hidden) - 1],
                                    len(self.problem.target_labels)))
        layers.append({
            'weights': M,
            'h': None,
            'h_derivate': None,
            'error': None
        })
        return layers

    def fowardPropagate(self, layers, row):
        input = row
        for layer in layers:
            layer['h'] = self.activation_function(
                np.dot(input, layer['weights']))
            layer['h_derivate'] = self.derivate(layer['h'])
            input = layer['h']
        return layers

    def backwardPropagate(self, layers, row, expected):
        for layer_index in reversed(range(len(layers))):
            layer = layers[layer_index]
            if (layer_index == len(layers) - 1):
                layer['error'] = expected - layer['h']
            else:
                next_layer = layers[layer_index + 1]
                layer['error'] = np.dot(
                    next_layer['weights'],
                    next_layer['h_derivate'] * next_layer['error'])
        return layers

    def updateWeights(self, layers, row, epoch):
        learn_rate = self.eta(epoch)
        input = row
        for layer_index, layer in enumerate(layers):
            prev_layer = layers[layer_index - 1]
            if layer_index != 0:
                input = prev_layer['h']
            layer['weights'] = layer['weights'] + learn_rate * (
                np.outer(input, layer['error']) * layer['h_derivate'])
        return layers

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, layers, train_df):
        train_data = train_df[self.problem.data_labels].values
        target = train_df[self.problem.target_labels].values

        for epoch in range(self.max_epochs):
            shuffle(train_data, target)
            for row, row_target in zip(train_data, target):
                layers = self.fowardPropagate(layers, row)
                layers = self.backwardPropagate(layers, row, row_target)
                layers = self.updateWeights(layers, row, epoch)

        return layers

    def hit_rate(self, layers, data):
        actual = data[self.problem.target_labels].values
        test = data[self.problem.data_labels].values
        hit = 0
        for row, row_target in zip(test, actual):
            layers = self.fowardPropagate(layers, row)
            y = self.validate(layers[len(layers) - 1]['h'])
            if (y == row_target).all():
                hit += 1
        return (hit / float(len(actual))) * 100.0

    def realize(self):
        train, test = train_test_split(self.problem.data, test_size=0.2)
        layers = self.initLayers()
        layers = self.train(layers, train)
        return train, test, self.hit_rate(layers, test)

    def evaluate(self, n=20):
        for _ in range(n):
            r = self.realize()
            self.realizations.append(r)
        hit_rates = np.array(self.realizations)[:, 2].astype(float)
        acc = np.average(hit_rates)
        std = np.std(hit_rates, dtype=np.float32)
        index = (np.abs(hit_rates)).argmax()
        print("Accuracy=%f, Standard deviation=%f" % (acc, std))

    def validate(self, prediction):
        return [
            1 if output == np.amax(prediction) else 0 for output in prediction
        ]

    def eta(self, epoch):
        return self.eta_initial * (self.eta_final /
                                   self.eta_initial)**(epoch / self.max_epochs)


mlp = MLP(problem=Vertebral(), n_hidden=[12], max_epochs=400)

# mlp = MLP(problem=Vertebral(),
#           eta_initial=0.5,
#           eta_final=0.1,
#           n_hidden=[12],
#           max_epochs=500)

mlp.evaluate(n=1)