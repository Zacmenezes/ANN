from abstract_problem import AbstractProblem
import numpy as np
import pandas as pd

class Toy(AbstractProblem):

    def __init__(self):
        pass

    def prepare_dataset(self, path):
        
        return df

    def generate_data(self):
       corners = [[0,0], [0,1], [1,0], [1,1]]
       data = []
       for corner in corners:
           for i in range(10):
                new = corner
                print(new)
                if(i < 5):
                    new[0] = corner[0] + (np.random.random_sample(1)/5)[0]
                else:
                    new[1] = corner[1] + (np.random.random_sample(1)/5)[0]
                data.append(new)
       return data

import random
import matplotlib.pyplot as plt

deviationFromPoint = 0.09

def create_points(source):
    points = []
    for _ in range(10):
        newCoords = [source[i] + random.random() * deviationFromPoint for i in range(2)]
        points.append(newCoords)
    return points

p = np.array(create_points([0,0]))
x = p[:,0]
y = p[:,1]
plt.scatter(x, y)

p = np.array(create_points([0,1]))
x = p[:,0]
y = p[:,1]
plt.scatter(x, y)

p = np.array(create_points([1,0]))
x = p[:,0]
y = p[:,1]
plt.scatter(x, y)

p = np.array(create_points([1,1]))
x = p[:,0]
y = p[:,1]
plt.scatter(x, y)

plt.show()