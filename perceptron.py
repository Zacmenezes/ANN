import numpy as np
import pandas as pd
from iris_problem import Iris
from toy_problem import Toy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, problem, l_rate, n_epoch):
        self.problem = problem
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.cms = []
        self.weights = []

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
        return (actual == np.array(predicted)).sum() / float(len(actual)) * 100.0

    def evaluate(self):
        hit_rates = []
        while(len(hit_rates) < 20):
            # train, test = self.split(self.problem.data, 0.8)
            train, test = train_test_split(self.problem.data, test_size=0.2, stratify=self.problem.data['d'])
            weights = self.train_weights(train, self.l_rate, self.n_epoch)
            hit_rates.append(self.hit_rate(test, weights))
            self.weights = weights
        acc = np.average(hit_rates)
        self.print_confusion_matrixes(hit_rates, acc)
        # self.plot_decision_surface()
        return acc, np.std(hit_rates)

    def plot_decision_surface(self):
        x = np.linspace(0,1.1,20)
        y = np.linspace(0,1.1,20)
        for i in x:
            for j in y:
                p = self.predict([1, i, j], self.weights)
                if(p == 1):
                    plt.scatter(i, j, c='red')
                else:
                    plt.scatter(i, j, c='blue')
        plt.show()

    def print_confusion_matrixes(self, hit_rates, acc):
        for i in range(len(hit_rates)):
            if(hit_rates[i] <= acc and hit_rates[i] < 100.0):
                cm = pd.DataFrame(self.cms[i])
                print("A/P \n%s, iteration=%d" %(cm, i))

    # def debug(self, hit_rates, cms):
    #     print("")
    #     for i in range(len(hit_rates)):
    #         print("Accuracy=%f, iteration=%d" %(hit_rates[i], i))
    #     print("")
    #     for i in range(len(cms)):
    #         print("%s, iteration=%d" %(cms[i], i))
    #     print("")

def main():
    
    setosa = Perceptron(Iris('Iris-setosa', 'data/iris.data'), 0.1, 20)
    print("Accuracy=%f, Standard deviation=%f" % setosa.evaluate())
    
    # versicolor = Perceptron(Iris('Iris-versicolor', 'data/iris.data'), 0.1, 20)
    # print("Accuracy=%f, Standard deviation=%f" % versicolor.evaluate())
    
    # virginica = Perceptron(Iris('Iris-virginica', 'data/iris.data'), 0.1, 20)
    # print("Accuracy=%f, Standard deviation=%f" % virginica.evaluate())

    # tp = Perceptron(Toy(), 0.1, 20)
    # Toy().plot_data()
    # print("Accuracy=%f, Standard deviation=%f" % tp.evaluate())
    
if __name__ == "__main__":
    main()