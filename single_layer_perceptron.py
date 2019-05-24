import numpy as np
import pandas as pd

class SingleLayerPerceptron():
 
    def __init__(self, learn_rate=0.1, neurons=3, max_epochs=200, problem):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.neurons = neurons

    def predict(self, row, weights):
        activation = np.dot(row, weights)
        return 1 if activation >= 0.0 else 0

    def train_weights(self, train_df, l_rate, n_epoch):
        # train_values = train_df.values
        # weights = np.ones((train_values.shape[1] - 1, self.neurons))
        # epoch = 0
        # while(epoch < n_epoch):
        #     np.random.shuffle(train_values)
        #     prediction = self.predict(row[:-3], weights)
        #     errors [row[-3] - prediction, row[-2] - prediction, row[-1] - prediction]
        #     weights
        #     for row in train_values:
        #         prediction = self.predict(row[:-3], weights)
        #         e1 = row[-1] - prediction
        #         e2 = row[-2] - prediction
        #         e3 = row[-3] - prediction
        #         weights = weights + l_rate * error * row[:-3]
        #     epoch += 1
        return weights