from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Iris(AbstractProblem):

    def __init__(self, label=None, drop=[]):
        self.label = label
        self.data = self.prepare_dataset('data/iris.data')
        if(len(drop) == 2):
            self.data = self.data.drop(drop, axis=1)    

    def prepare_dataset(self, path):
        df = pd.read_csv(path, header=None)
        df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
        df = self.map_class(df)    
        df = self.create_x0(df).join(df)
        return df
    
    def map_class(self, df):
        if(self.label == None):
            df['d0'] = np.where(df['d'] == 'Iris-setosa', 1, 0)
            df['d1'] = np.where(df['d'] == 'Iris-versicolor', 1, 0)
            df['d2'] = np.where(df['d'] == 'Iris-virginica', 1, 0)
            df = df.drop(['d'], axis=1)
        else:
            df['d'] = np.where(df['d'] == self.label, 0, 1)
        return df


def degrau(valor):
    return 1 if valor >=0 else 0

iris = Iris()
data = iris.data.drop(['d0','d1','d2'], axis=1)[90:100].values
target = iris.data[['d0','d1','d2']][90:100].values

x = np.array([data[0], data[0], data[0]])
w = np.ones((5,3))
etha = 0.1
predictions = np.dot(x, w)

vfunc = np.vectorize(degrau)
predictions = vfunc(predictions)

errors = target[0] - predictions
print(predictions, errors)

w = w + etha * np.outer(data[0], errors.T[0])
print(w)