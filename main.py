from random import randint

import numpy as np

from utils.dataset import Dataset
from utils.neuralnetwork import NeuralNetwork

# La red está memorizando, no está aprendiendo :/ (overfitting)

ds = Dataset('dataset/data.csv')
dataX, dataY = ds.getData()
test = 1000
X_training = dataX[:int(len(dataX)*0.9)]
Y_training = dataY[:int(len(dataX)*0.9)]

X_test = dataX[-int(len(dataX)*0.1):]
Y_test = dataY[-int(len(dataX)*0.1):]

features = len(X_training[0])
classes = len(Y_training[0])

nn = NeuralNetwork([features, 50, classes],activations=['relu', 'sigmoid'])
nn.train(X_training.T, Y_training.T, epochs=1000, batch_size=64, lr = .05)

correct = 0
incorrect = 0
for i in range(test):
    randomValue = randint(0, len(X_test) - 1)

    x = np.array([X_test[randomValue]]).T
    y = np.argmax(Y_test[randomValue])
    z_s, a_s = nn.feedforward(x)

    y_pred = np.argmax(a_s[-1])
    if y_pred == y:
        correct +=1

acc = (correct/test) * 100

print(f"Accurate: {acc}%")

