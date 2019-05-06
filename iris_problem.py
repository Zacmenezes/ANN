from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Iris(AbstractProblem):

    def __init__(self, label, dataset_path, drop=[]):
        self.label = label
        self.data = self.prepare_dataset(dataset_path)
        if(len(drop) == 2):
            self.data = self.data.drop(drop, axis=1)    

    def prepare_dataset(self, path):
        df = pd.read_csv(path, header=None)
        df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
        df['d'] = np.where(df['d']==self.label, 0, 1)
        x0 = self.create_x0(df)
        df = x0.join(df)
        return df

