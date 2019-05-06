import numpy as np
import pandas as pd
from iris_problem import Iris
from toy_problem import Toy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn


class Perceptron:

    index = 0
    def __init__(self, problem, l_rate, n_epoch):
        self.problem = problem
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.cms = []
        self.weights = []
        self.test = []

    def predict(self, row, weights):
        activation = np.dot(row, weights)
        return 1 if activation >= 0.0 else 0

    def train_weights(self, train_df, l_rate, n_epoch):
        train_values = train_df.values
        weights = np.random.randn(len(train_df.columns) - 1)
        epoch = 0
        while(epoch < n_epoch):
            np.random.shuffle(train_values)
            for row in train_values:
                prediction = self.predict(row[:-1], weights)
                error = row[-1] - prediction
                weights = weights + l_rate * error * row[:-1]
            epoch += 1
        return weights

    def split(self, data, frac):
        aux = data.sample(frac=frac)
        return aux, data.drop(aux.index)

    def hit_rate(self, data, trained_weights):
        actual = data['d'].values
        predicted = []
        cm = np.zeros((2, 2))
        for row in data.values:
            predicted.append(self.predict(row[:-1], trained_weights)) 
        for a, p in zip(actual, predicted):
            cm[a][p] += 1
        self.cms.append(cm)
        self.weights.append(trained_weights)
        self.test.append(data)
        return (actual == np.array(predicted)).sum() / float(len(actual)) * 100.0

    def evaluate(self):
        hit_rates = []
        while(len(hit_rates) < 20):
            train, test = train_test_split(self.problem.data, test_size=0.2, stratify=self.problem.data['d'])
            weights = self.train_weights(train, self.l_rate, self.n_epoch)
            hit_rates.append(self.hit_rate(test, weights))
        acc = np.average(hit_rates)
        index = (np.abs(hit_rates-acc)).argmin()      
        # self.plot_confusion_matrixes(index)
        # self.plot_decision_surface(index)

        return acc, np.std(hit_rates)

    def plot_decision_surface(self, index):
        test = self.test[index]
        max_y = test[test['x1'] == test['x1'].max()]['x1'].values[0]
        max_x = test[test['x2'] == test['x2'].max()]['x2'].values[0]
        m = max(max_x, max_y)
        
        y = np.linspace(0, m, 100)
        x = np.linspace(0, m, 100)
        # for i in x:
        #     for j in y:
        #         p = self.predict([1, i, j], self.weights[index])
        #         if(p == 1):
        #             plt.scatter(i, j, c='#FF6666', marker='.')
        #         else:
        #             plt.scatter(i, j, c='#72bcd4', marker='.')

        for row in test.values:
            p = self.predict(row[:-1], self.weights[index])
            if(p == 1):
                plt.scatter(row[1], row[2], c='red', marker='v')
            else:
                plt.scatter(row[1], row[2], c='blue', marker='*')

    def plot_confusion_matrixes(self, index):
        array = self.cms[index]        
        df_cm = pd.DataFrame(array, range(2), range(2))
        plt.figure(figsize = (2,2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})

def main():  
    # setosa = Perceptron(Iris('Iris-setosa', 'data/iris.data'), 0.01, 300)
    # print("Accuracy=%f, Standard deviation=%f" % setosa.evaluate())
    # versicolor = Perceptron(Iris('Iris-versicolor', 'data/iris.data'), 0.1, 20)
    # print("Accuracy=%f, Standard deviation=%f" % versicolor.evaluate())
    
    # virginica = Perceptron(Iris('Iris-virginica', 'data/iris.data'), 0.01, 300)
    # print("Accuracy=%f, Standard deviation=%f" % virginica.evaluate())

    tp = Perceptron(Toy(), 0.1, 100)
    print("Accuracy=%f, Standard deviation=%f" % tp.evaluate())
    plt.show()
  
    
if __name__ == "__main__":
    main()