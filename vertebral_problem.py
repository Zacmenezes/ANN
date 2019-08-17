from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd


class Vertebral(AbstractProblem):
    def __init__(self, drop=[]):
        self.data = self.prepare_dataset('data/vertebral.data')
        if (len(drop) == 2):
            self.data = self.data.drop(drop, axis=1)

    def prepare_dataset(self, path):
        df = pd.read_csv(path, header=None)
        df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'd']
        df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']] = df[['x1', 'x2', 'x3', 'x4', 'x5','x6']].apply(self.normalize)
        df = self.map_class(df)
        df = self.create_x0(df).join(df)
        return df

    def map_class(self, df):
       
        df['d0'] = np.where(df['d'] == 'Hernia', 1, 0)
        df['d1'] = np.where(df['d'] == 'Spondylolisthesis', 1, 0)
        df['d2'] = np.where(df['d'] == 'Normal', 1, 0)
        df = df.drop(['d'], axis=1)
    
        return df

    def normalize(self, df):
        return (df - df.min()) / (df.max() - df.min())
