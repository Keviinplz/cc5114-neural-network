# Importaciones Nativas
import re
import random

# Importaciones de Librerias
import numpy as np
from sklearn.model_selection import KFold

# Constantes
FILE_PATH = 'dataset/data.csv'
REGULAR_EXPRESION = r'^(\w[#b]|\w);(.+);".+";((\w.){5}\w);.+$'
seed = 42

# Abrimos el dataset y lo guardamos en un string
with open(FILE_PATH, 'r') as file:
    file_to_string = file.read()

# Usando expresiones regulares, extraemos la información relevante del dataset
dirty_data = re.findall(REGULAR_EXPRESION, file_to_string, re.MULTILINE)

# Limpiamos los datos para darle un formato de tipo
# dato = [input, output]

def mapping(n):
    '''

    '''
    return -1 if n == 'x' else int(n)

clean_data = []
for data in dirty_data:
    inputData = data[2].split(',')
    updateInputData = list(map(mapping, inputData))
    output = data[0]
    cosa = [updateInputData, output]
    clean_data.append(cosa)

X = []
outputData = []
for data in clean_data:
    X.append(data[0])
    outputData.append(data[1])

for data in X:
    for string in data:
        if string == 'x':
            string = -1

bass = []
for y in outputData:
    if y not in bass:
        bass.append(y)

Y = []
for data in outputData:
    y = np.zeros(len(bass))
    for i in range(len(bass)):
        if data == bass[i]:
            y[i] = 1
            Y.append(y)

# ordinariez
X = np.array(X)
Y = np.array(Y)
#kf = KFold(n_splits=10, shuffle=True, random_state=seed)
#for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    Y_train, Y_test = Y[train_index], Y[test_index]
    

# 2632/10 -> 263
# sea X una caja (son 10) -> Entrenar con Y cajas, con Y != X
# Testear con X
# Calcular Precision y Recall
# Repetir con el otro grupo
###########################################################
def sigmoid(z):
	return 1/(1 + np.exp(-z))

# Produce a neural network randomly initialized
def initialize_parameters(n_x, n_h, n_y):
	W1 = np.random.randn(n_h, n_x)
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h)
	b2 = np.zeros((n_y, 1))

	parameters = {
	"W1": W1,
	"b1" : b1,
	"W2": W2,
	"b2" : b2
	}
	return parameters

# Evaluate the neural network
def forward_prop(X, parameters):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  # Z value for Layer 1
  Z1 = np.dot(W1, X) + b1
  # Activation value for Layer 1
  A1 = np.tanh(Z1)
  # Z value for Layer 2
  Z2 = np.dot(W2, A1) + b2
  # Activation value for Layer 2
  A2 = sigmoid(Z2)

  cache = {
    "A1": A1,
    "A2": A2
  }
  return A2, cache

# Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y 
# We use the Mean Square Error cost function
def calculate_cost(A2, Y):
  # m is the number of examples
  cost = np.sum((0.5 * (A2 - Y) ** 2).mean(axis=1))/m
  return cost

# Apply the backpropagation
def backward_prop(X, Y, cache, parameters):
  A1 = cache["A1"]
  A2 = cache["A2"]

  W2 = parameters["W2"]

  # Compute the difference between the predicted value and the real values
  dZ2 = A2 - Y
  dW2 = np.dot(dZ2, A1.T)/m
  db2 = np.sum(dZ2, axis=1, keepdims=True)/m
  # Because d/dx tanh(x) = 1 - tanh^2(x)
  dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
  dW1 = np.dot(dZ1, X.T)/m
  db1 = np.sum(dZ1, axis=1, keepdims=True)/m

  grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
  }

  return grads

# Third phase of the learning algorithm: update the weights and bias
def update_parameters(parameters, grads, learning_rate):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  dW1 = grads["dW1"]
  db1 = grads["db1"]
  dW2 = grads["dW2"]
  db2 = grads["db2"]

  W1 = W1 - learning_rate*dW1
  b1 = b1 - learning_rate*db1
  W2 = W2 - learning_rate*dW2
  b2 = b2 - learning_rate*db2
  
  new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
  }

  return new_parameters

# model is the main function to train a model
# X: is the set of training inputs
# Y: is the set of training outputs
# n_x: number of inputs (this value impacts how X is shaped)
# n_h: number of neurons in the hidden layer
# n_y: number of neurons in the output layer (this value impacts how Y is shaped)
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
  parameters = initialize_parameters(n_x, n_h, n_y)
  for i in range(0, num_of_iters+1):
    a2, cache = forward_prop(X, parameters)
    cost = calculate_cost(a2, Y)
    grads = backward_prop(X, Y, cache, parameters)
    parameters = update_parameters(parameters, grads, learning_rate)
    if(i%100 == 0):
      print('Cost after iteration# {:d}: {:f}'.format(i, cost))

  return parameters


# Make a prediction
# X: represents the inputs
# parameters: represents a model
# the result is the prediction
def predict(X, parameters):
  a2, cache = forward_prop(X, parameters)
  yhat = a2
  yhat = np.squeeze(yhat)
  yhat = softmax(yhat)
  return np.where(yhat == np.amax(yhat))

# Set the seed to make result reproducible
np.random.seed(seed)

# No. of training examples
m = 2632

# Set the hyperparameters
n_x = 6      #No. of neurons in first layer
n_h = 20     #No. of neurons in hidden layer
n_y = 18     #No. of neurons in output layer

#The number of times the model has to learn the dataset
number_of_iterations = 10000
learning_rate = 0.01

# define a model 
trained_parameters = model(X.T, Y.T, n_x, n_h, n_y, number_of_iterations, learning_rate)

X_test = np.array([0, 3, 2, 0, 1, 0])
Y_predict = np.zeros(len(Y))
Y_predict[13] = 1

output = predict(X_test, trained_parameters)
print(output)