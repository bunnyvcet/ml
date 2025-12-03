import numpy as np
from minisom import MiniSom
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
target = iris.target

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, num_iteration=100)

plt.figure(figsize=(7, 7))
for i, x in enumerate(data):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, str(target[i]),
             color=plt.cm.Set1(target[i] / 3),
             fontdict={'weight': 'bold', 'size': 11})

plt.title('SOM clustering of Iris data')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.grid()
plt.show()
