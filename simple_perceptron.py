import numpy as np
import pandas as pd

# Make a prediction with weights
def predict(row, weights):
    activation = np.dot(row, weights)
    return 1 if activation >= 0.0 else 0

def train_weights(train, l_rate, n_epoch):
    train_values = train.values
    weights = np.ones(len(train.columns) - 2)
    weights = np.append(-1, weights)
    for epoch in range(n_epoch):
        for row in train_values:
            prediction = predict(row[:-1], weights)
            error = row[-1] - prediction
            weights = weights + l_rate * error * row[:-1]
    return weights

def create_x0(data):
    return pd.DataFrame({'x0': np.ones(len(data.index))})

# Preparing iris dataset
iris_df = pd.read_csv("data/iris.data", header=None)
iris_df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
iris_df['d'] = iris_df['d'].map({'Iris-setosa': 0,'Iris-versicolor': 1, 'Iris-virginica': 1})
iris_df['label'] = iris_df['d'].map({0 : 'Setosa', 1 : 'Other'})
x0 = create_x0(iris_df)
iris_df = x0.join(iris_df)

# Split between train and test
train=iris_df.sample(frac=0.8,random_state=250)
test=iris_df.drop(train.index)

train = train.drop(columns=['label'])

w = train_weights(train, 0.2, 20)

# print(test.values)
for row in test.values:
	prediction = predict(row[:-2], w)
	print("Expected=%d, Predicted=%d" % (row[-2], prediction))

# x = predict([1, 5.3,3.7,1.5,0.2], w)
# print(x)