from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Toy2(AbstractProblem):
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.data = self.prepare_dataset('')

    def prepare_dataset(self, path):
        X = np.linspace(0,10,500)
        Y = [(self.a * x + self.b) + np.random.uniform(-5,5) for x in X]
        df = pd.DataFrame(np.array([[i,j] for i,j in zip(X,Y)]))
        df.columns = ['x','y']
        x0 = self.create_x0(df)
        df = x0.join(df)
        print(df)
        return df
