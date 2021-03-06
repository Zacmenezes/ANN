from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Toy2(AbstractProblem):
    
    def __init__(self, a, b, c=None):
        self.a = a
        self.b = b
        self.c = c
        self.data = self.prepare_dataset('')

    def prepare_dataset(self, path):
        if(self.c != None):
            return self.create_3d()
        return self.create_2d()
        
    def create_2d(self):
        X = np.linspace(0, 10, 500)
        Y = [(self.a * x + self.b) + np.random.uniform(-2,2) for x in X]
        df = pd.DataFrame(np.array([[i,j] for i,j in zip(X,Y)]))
        df.columns = ['x','y']
        df[['x']] = df[['x']].apply(self.normalize)
        x0 = self.create_x0(df)
        df = x0.join(df)
        return df

    def create_3d(self):
        X = np.random.rand(100)
        Y = np.random.rand(100)
        Z = [( (self.a * x) + (self.b * y) + self.c) + np.random.uniform(-1, 1) for x, y in zip(X, Y)]
        df = pd.DataFrame(np.array([[X[i], Y[i], Z[i]] for i in range(len(Z))]))
        df.columns = ['x','y','z']
        df[['x', 'y']] = df[['x', 'y']].apply(self.normalize)
        x0 = self.create_x0(df)
        df = x0.join(df)
        return df

    def normalize(self, df):
        return (df-df.min())/(df.max()-df.min())
