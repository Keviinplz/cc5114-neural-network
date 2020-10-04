from random import randint, shuffle

import numpy as np

from utils.dataset import Dataset
from utils.neuralnetwork import NeuralNetwork
from utils.otherNeuralNetwork import OtherNeuralNetwork

# Create a Training and Test data by factor
# inputData: list Data input
# outputData: list Data output
# factor: float between 0 and 1, to separate data
def createTestAndTrainingDataset(inputData: list, outputData: list, factor: float) -> list:
    training_len = int(len(inputData)*factor)
    test_len = -int(len(inputData)*(1-factor))
    
    X_training = inputData[:training_len]
    Y_training = outputData[:training_len]

    X_test = inputData[test_len:]
    Y_test = outputData[test_len:]

    return X_training, Y_training, X_test, Y_test

def train(X_training, Y_training, X_test, Y_test, n_h, epoch, learning_rate):
    n_x = len(X_training[0])
    n_y = len(Y_training[0])

    nn = OtherNeuralNetwork(X_training.T, Y_training.T, n_x, n_h, n_y, epoch, learning_rate)

    trained_params = nn.model()

    accurateModel(X_test, Y_test, nn, trained_params)
 
def accurateModel(X_test: list, Y_test: list, nn: OtherNeuralNetwork, trained_params: list) -> None:
    correct = 0
    for i in range(len(X_test)):
        randomValue = randint(0, (len(X_test) - 1))

        x = np.array([X_test[randomValue]]).T
        y = Y_test[randomValue][0]

        y_pred = nn.predict(x, trained_params)
        if y_pred == y:
            correct += 1
    acc = (correct/len(X_test)) * 100

    print(f'Accurate: {acc}%')

if __name__ == '__main__':

    FILEPATH = 'dataset/stars01.csv'
    REGEX = r'^(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)([\w/.: \-\+\(\)]+,)(\d+\.\d+,|-\d+\.\d+,)([01])$'
    GROUP = [1, 2, 3, 4, 6, 7]

    ds = Dataset(FILEPATH, REGEX, GROUP)

    X, Y = ds.getData()

    X_train, Y_train, X_test, Y_test = createTestAndTrainingDataset(X, Y, 0.8)
    train(X_train, Y_train, X_test, Y_test, 50, 5000, 0.01)
