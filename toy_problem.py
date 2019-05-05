from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Toy(AbstractProblem):
    
    deviationFromPoint = 0.09

    def __init__(self):
        self.data = self.prepare_dataset("data/toy.data")

    def prepare_dataset(self, path):
        data = pd.read_csv(path, sep=",")
        x0 = self.create_x0(data)
        data = x0.join(data)
        return data

    def generate_data(self):
        data = self.create_points([0,1], 0)
        data = data.append(self.create_points([1,0], 0))
        data = data.append(self.create_points([0,0], 0))
        data = data.append(self.create_points([1,1], 1))
        return data

    def create_points(self, source, _class):
        points = []
        for _ in range(10):
            coords = [source[i] + random.random() * self.deviationFromPoint for i in range(2)]
            coords.append(_class)
            points.append(coords)          
        return pd.DataFrame(data=points, columns=['x1', 'x2', 'd'])

    def plot_data(self):
        c0 = self.data.loc[self.data['d'] == 0]
        c1 = self.data.loc[self.data['d'] == 1]

        plt.scatter(c0['x1'], c0['x2'], c='red')
        plt.scatter(c1['x1'], c1['x2'])

        plt.show()

# t = Toy()
# t.plot_data()