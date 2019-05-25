from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Toy(AbstractProblem):
    
    deviationFromPoint = 0.5

    def __init__(self, neurons=1, class_size=10):
        self.neurons = neurons
        self.class_size = class_size
        self.data = self.prepare_dataset("data/toy.data")

    def prepare_dataset(self, path):
        data = self.generate_data()
        data.columns = ['x1', 'x2', 'd']
        data = self.map_class(data)
        x0 = self.create_x0(data)
        data = x0.join(data)
        return data

    def generate_data(self):
        data = self.create_points([1,1], 'a')
        data = data.append(self.create_points([1,0], 'b'),  ignore_index=True)
        data = data.append(self.create_points([0,0], 'c'),  ignore_index=True)
        if self.neurons == 1:
            data = data.append(self.create_points([0,1], 'd'), ignore_index=True)
        return data

    def create_points(self, source, _class):
        points = []
        for _ in range(self.class_size):
            coords = [source[i] + random.random() * self.deviationFromPoint for i in range(2)]
            coords.append(_class)
            points.append(coords)          
        return pd.DataFrame(data=points, columns=['x1', 'x2', 'd'])

    def map_class(self, df):
        if(self.neurons == 3):
            df['d0'] = np.where(df['d'] == 'a', 1, 0)
            df['d1'] = np.where(df['d'] == 'b', 1, 0)
            df['d2'] = np.where(df['d'] == 'c', 1, 0)
            df = df.drop(['d'], axis=1)
        else:
            df['d'] = np.where(df['d'] == 'a', 0, 1)
        return df