from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Breast(AbstractProblem):

    def __init__(self, drop=[]):
        self.data_labels = []
        self.target_labels = []
        self.data = self.prepare_dataset('data/breast-cancer-wisconsin.data')
        if(len(drop) > 0):
            self.data = self.data.drop(drop, axis=1)    

    def prepare_dataset(self, path):
        df = pd.read_csv(path, header=None)
        df = df.drop(df.columns[0], axis=1) # remove sample id
        cols = self.get_column_labels(df) + ['d']
        df.columns = cols
        df = self.map_class(df)
        self.data_labels = df.columns[:9].tolist()
        self.target_labels = df.columns[9:].tolist()
        df[self.data_labels] = df[self.data_labels].apply(self.normalize)
        df = self.create_x0(df).join(df)
        self.data_labels = ['x0'] + self.data_labels
        
        return df
    
    def get_column_labels(self, df):
        length = df.values.shape[1] - 1
        labels = ['x' + str(i + 1) for i in range(length)]
        return labels

    def map_class(self, df):
        df['d0'] = np.where(df['d'] == 2, 1, 0)
        df['d1'] = np.where(df['d'] == 4, 1, 0)
        df = df.drop(['d'], axis=1)

        return df

    def normalize(self, df):
        return (df-df.min())/(df.max()-df.min())

# b = Breast()
# print(b.data.tail())