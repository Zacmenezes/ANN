import numpy as np
import pandas as pd

def predict(row, weights):
    activation = np.dot(row, weights)
    return 1 if activation >= 0.0 else 0

def train_weights(train, l_rate, n_epoch):
    train_values = train.values
    weights = np.append( -1 , np.ones(len(train.columns) - 2))
    for epoch in range(n_epoch):
        for row in train_values:
            prediction = predict(row[:-1], weights)
            error = row[-1] - prediction
            weights = weights + l_rate * error * row[:-1]
    return weights

def create_x0(data):
    return pd.DataFrame({'x0': np.ones(len(data.index))})

def prepare_dataset(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
    df['d'] = df['d'].map({'Iris-setosa': 1,'Iris-versicolor': 0, 'Iris-virginica': 0})
    x0 = create_x0(df)
    df = x0.join(df)
    return df

def hit_rate(data):
    hit = 0
    for row in data:
        actual = row[-1]
        prediction = predict(row[:-1], w)
        if actual == prediction:
            hit += 1
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))
    return hit / float(len(data)) * 100.0

iris_df = prepare_dataset("data/iris.data")
train=iris_df.sample(frac=0.8)
test=iris_df.drop(train.index)
w = train_weights(train, 0.1, 20)

print(hit_rate(test.values))