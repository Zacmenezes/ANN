from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class XOR(AbstractProblem):
    
    deviationFromPoint = 0.1

    def __init__(self, class_size=50):
        self.class_size = class_size
        self.data_labels = []
        self.target_labels = []
        self.data = self.prepare_dataset()
        # self.plot_data()

    def prepare_dataset(self):
        df = self.generate_data()
        cols = self.get_column_labels(df) + ['d']
        df.columns = cols
        self.data_labels = df.columns[:2].tolist()
        self.target_labels = df.columns[2:].tolist()
        df[self.data_labels] = df[self.data_labels].apply(self.normalize)
        df = self.create_x0(df).join(df)
        self.data_labels = ['x0'] + self.data_labels

        return df

    def get_column_labels(self, df):
        length = df.values.shape[1] - 1
        labels = ['x' + str(i + 1) for i in range(length)]
        return labels

    def generate_data(self):
        data = self.create_points([0,0], 0)
        data = data.append(self.create_points([0,1], 1), ignore_index=True)
        data = data.append(self.create_points([1,1], 0), ignore_index=True)
        data = data.append(self.create_points([1,0], 1),  ignore_index=True)
        return data

    def create_points(self, source, _class):
        points = []
        for _ in range(self.class_size):
            coords = [source[i] + random.random() * self.deviationFromPoint for i in range(2)]
            coords.append(_class)
            points.append(coords)          
        return pd.DataFrame(data=points, columns=['x1', 'x2', 'd'])


    def plot_data(self):
        target = self.data[['d']].values
        data = self.data.drop(['d'], axis=1).values

        fig, ax = plt.subplots()
        ax.set_facecolor((0.97, 0.97, 0.97))
        for row, row_target in zip(data, target):
            if (row_target == np.array([1])).all():
                ax.scatter(row[1], row[2], c='red', marker='o')
            elif (row_target == np.array([0])).all():
                ax.scatter(row[1], row[2], c='blue', marker='*')   

        plt.show()

    def normalize(self, df):
        return (df-df.min())/(df.max()-df.min())

