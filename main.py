from random import randint, shuffle

import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import Dataset
from utils.neuralnetwork import NeuralNetwork
from utils.otherNeuralNetwork import OtherNeuralNetwork
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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

def createNeuralNetwork(n_x: int, n_h: int, n_y: int, epoch: int, learning_rate: float):
    return OtherNeuralNetwork(n_x, n_h, n_y, epoch, learning_rate)

def train(X_training, Y_training, nn):

    #nn = OtherNeuralNetwork(X_training.T, Y_training.T, n_x, n_h, n_y, epoch, learning_rate)
    trained_params = nn.model(X_training.T, Y_training.T)
    return trained_params
    
  
 
def accurateModel(X_test: list, Y_test: list, nn: OtherNeuralNetwork, trained_params: list) -> None:
    
    correct = 0
    for i in range(len(X_test)):
        randomValue = randint(0, (len(X_test) - 1))

        x = np.array([X_test[randomValue]]).T
        y = Y_test[randomValue][0]

        y_pred = nn.predict(x, trained_params)

        #confusion matrix
        
        if y_pred == y:
            correct += 1
    acc = (correct/len(X_test)) * 100

    print('Accurate: {:.2f}%'.format(acc))

def confusionMatrix(X_test: list, Y_test: list, nn: OtherNeuralNetwork, trained_params: list, labels: list, plot: bool):
    '''
    This function receives the test values to contruct the confusion matrix.
    Returns the above mentioned matrix (M) and if plot = True this function plot the matrix
    '''
    Y_pred = []
    for i in range(len(X_test)):
        x = np.array([X_test[i]]).T
        y_pred =  nn.predict(x, trained_params)
        Y_pred.append(y_pred)
   
    M = confusion_matrix(Y_test, Y_pred)
    
    if plot:

        fig, ax = plt.subplots()
        ax.imshow(M, cmap=plt.cm.Blues)
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        for i in range(len(labels)):
            for j in range(len(labels)):
                c = M[j,i]
                ax.text(i, j, str(c), va='center', ha='center')
        plt.show()
    return M

def precision(M, labels):
    '''
    This function takes the confusion matrix and prints the precision values
    '''
    row_sum = [sum(row[i] for i in range(len(M[0]))) for row in M]
    precision = [M[i][i]/row_sum[i] for i in range(len(row_sum))]
    
    for i in range(len(M[0])):
        print(f"Recall of {labels[i]} is " + "{:.2f}%".format(precision[i]*100))

def recall(M, labels):
    '''
    This function takes the confusion matrix and prints the recall values
    '''
    precision = [M[i][i]/sum(row[i] for row in M) for i in range(len(M[0]))]
    
    for i in range(len(M[0])):
        print(f"Recall of {labels[i]} is " + "{:.2f}%".format(precision[i]*100)) 

if __name__ == '__main__':

    FILEPATH = 'dataset/stars01.csv'
    REGEX = r'^(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)(\d+\.\d+,|-\d+\.\d+,)([\w/.: \-\+\(\)]+,)(\d+\.\d+,|-\d+\.\d+,)([01])$'
    GROUP = [1, 2, 3, 4, 6, 7]

    ds = Dataset(FILEPATH, REGEX, GROUP)

    X, Y = ds.getData()

    # X_train, Y_train, X_test, Y_test = createTestAndTrainingDataset(X, Y, 0.8)
    # train(X_train, Y_train, X_test, Y_test, 50, 1000, 0.01)

    # Creating a Neural Network
    n_x = len(X[0])
    n_y = len(Y[1])
    nn =  createNeuralNetwork(n_x, 50, n_y, 1000, 0.01)
    

    X_train, Y_train, X_test, Y_test = createTestAndTrainingDataset(X, Y, 0.8)
    trained_params = train(X_train, Y_train, nn)
    accurateModel(X_test, Y_test, nn, trained_params)

    #K-fold training method
    # kf = KFold(n_splits=5, shuffle=True, random_state = 123)
    # for train_index, test_index in kf.split(X):
    #     print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]

    #     trained_params = train(X_train, Y_train, nn)
    #     accurateModel(X_test, Y_test, nn, trained_params)
    
    #Confusion Matrix 
    labels = ["Dwarfs", "Giant"]
    M = confusionMatrix(X_test, Y_test, nn, trained_params, labels, plot = True)
    
    #Precision and Recall
    precision(M, labels)
    recall(M, labels)
