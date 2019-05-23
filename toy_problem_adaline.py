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
        X = np.linspace(0,10,500)
        if(self.c != None):
            return self.create_3d(X)
        return self.create_2d(X)
        
    def create_2d(self, X):
        Y = [(self.a * x + self.b) + np.random.uniform(-2,2) for x in X]
        df = pd.DataFrame(np.array([[i,j] for i,j in zip(X,Y)]))
        df.columns = ['x','y']
        df = self.normalize(df)
        x0 = self.create_x0(df)
        df = x0.join(df)
        return df

    def create_3d(self, X):
        Y = np.random.rand(100, 1)
        Z = [(self.a * x + self.b * y + self.c) + np.random.uniform(-0.1, 0.1) for x, y in zip(X, Y)]
        df = pd.DataFrame(np.array([[Z[i][0], X[i], Y[i][0]] for i in range(len(Z))]))
        df.columns = ['x','y','z']
        df = self.normalize(df)
        x0 = self.create_x0(df)
        df = x0.join(df)
        return df

    def normalize(self, df):
        return (df-df.min())/(df.max()-df.min())
