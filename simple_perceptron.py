import numpy as np
import pandas as pd

# Make a prediction with weights
def predict(row, weights):
    row = np.append([1], row[:2])
    activation = np.dot(row, weights)
    return 1 if activation >= 0.0 else 0

# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

# for row in dataset:
# 	prediction = predict(row, weights)
# 	print("Expected=%d, Predicted=%d" % (row[-1], prediction))

iris_df = pd.read_csv("data/iris.data", header=None)
iris_df.columns = ['x1', 'x2', 'x3', 'x4', 'label']

iris_df['label'] = iris_df['label'].map({'Iris-setosa': 'Setosa','Iris-versicolor': 'Other', 'Iris-virginica': 'Other'})

print(iris_df.head())
print(iris_df.tail())