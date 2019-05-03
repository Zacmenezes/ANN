import numpy as np
import pandas as pd

def predict(row, weights):
    activation = np.dot(row, weights)
    return 1 if activation >= 0.0 else 0

def train_weights(train, l_rate, n_epoch):
    train_values = train.values
    weights = np.append(-1 , np.ones(len(train.columns) - 2))
    for epoch in range(n_epoch):
        np.random.shuffle(train_values)
        for row in train_values:
            prediction = predict(row[:-1], weights)
            error = row[-1] - prediction
            weights = weights + l_rate * error * row[:-1]
    return weights

def create_x0(data):
    return pd.DataFrame({'x0': np.ones(len(data.index))})

def prepare_dataset(path, index):
    df = pd.read_csv(path, header=None)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'd']
    # df.drop(['x3', 'x4'], axis=1)
    df['d'] = np.where(df['d']==index, 0, 1)
    x0 = create_x0(df)
    df = x0.join(df)
    return df

def split(data, frac):
    aux = data.sample(frac=frac)
    return aux, data.drop(aux.index)

def hit_rate(data, trained_weights):
    hit = 0
    for row in data:
        actual = row[-1]
        prediction = predict(row[:-1], trained_weights)
        if actual == prediction:
            hit += 1
        # print("Expected=%d, Predicted=%d" % (row[-1], prediction))
    return hit / float(len(data)) * 100.0

def accuracy(data):
    results = []
    for i in range(20):
        train, test = split(data, 0.8)
        w = train_weights(train, 0.1, 20)
        results.append(hit_rate(test.values, w))
    return np.average(results)

setosa_df = prepare_dataset("data/iris.data", "Iris-setosa")
versicolor_df = prepare_dataset("data/iris.data", "Iris-versicolor")
virginica_df = prepare_dataset("data/iris.data", "Iris-virginica")

print(accuracy(setosa_df))
print(accuracy(versicolor_df))
print(accuracy(virginica_df))