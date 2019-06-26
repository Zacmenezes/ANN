from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Iris(AbstractProblem):

    def __init__(self, label=None, drop=[], inhibit=0):
        self.label = label
        self.inhibit = inhibit
        self.data = self.prepare_dataset('data/iris.data')
        if(len(drop) == 2):
            self.data = self.data.drop(drop, axis=1)    

    def prepare_dataset(self, path):
        df = pd.read_csv(path, header=None)
        df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
        df[['x1', 'x2', 'x3', 'x4']] = df[['x1', 'x2', 'x3', 'x4']].apply(self.normalize)
        df = self.map_class(df)    
        df = self.create_x0(df).join(df)
        return df
    
    def map_class(self, df):
        if(self.label == None):
            df['d0'] = np.where(df['d'] == 'Iris-setosa', 1, self.inhibit)
            df['d1'] = np.where(df['d'] == 'Iris-versicolor', 1, self.inhibit)
            df['d2'] = np.where(df['d'] == 'Iris-virginica', 1, self.inhibit)
            df = df.drop(['d'], axis=1)
        else:
            df['d'] = np.where(df['d'] == self.label, 0, 1)
        return df

    def normalize(self, df):
        return (df-df.min())/(df.max()-df.min())