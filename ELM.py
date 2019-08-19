import math
import numpy as np
from iris_problem import Iris
from vertebral_problem import Vertebral
from dermatology_problem import Dermatology
from breast_problem import Breast
from xor_problem import XOR
from sklearn.model_selection import train_test_split

class ELM(object):
    def __init__ (self, n_hidden=3, problem=Iris()):
        self.hid_num = n_hidden
        self.problem = problem
        self.out_num = len(problem.target_labels)

    def train(self, dataset):
        X, y = dataset[self.problem.data_labels], dataset[self.problem.target_labels]

        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))
        
        _H = np.linalg.pinv(self.sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)
        
        return self

    def test(self, test_data):
        X, target = test_data[self.problem.data_labels].values, test_data[self.problem.target_labels].values
        _H = self.sigmoid(np.dot(self.W, X.T))
        
        y = np.dot(_H.T, self.beta)
        y_label = [self.predict(out) for out in y]
    
        hits = 0
        for i,j in zip(y_label, target):
            if np.array_equal(i, j):
                hits += 1 
        
        return (hits / len(test_data)) * 100

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, outputs):
        return [1 if output == np.amax(outputs) else 0 for output in outputs]

problem = Dermatology()
train, test = train_test_split(problem.data, test_size=0.2)
e = ELM(n_hidden=12, problem=problem)
print(e.train(train).test(test))