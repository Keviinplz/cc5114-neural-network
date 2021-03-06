import numpy as np
from sklearn.metrics import mean_squared_error 

class NeuralNetwork: 
    """
    Define a Neural Network class
    """

    def __init__(self, n_x, n_h, n_y, number_of_iterations, learning_rate):
        """Constructor 

        Args:
            n_x (int): number of neurons in the first layer (input)
            n_h (int]): number of neurons in the hidden layer
            n_y (type): number of neuron in the last layer (output)
            number_of_iterations (int): times the neural network is trained
            learning_rate (float): rate of learning
        """
        # self.X = X
        # self.Y = Y
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
    

    @staticmethod
    def sigmoid(z):
        """ sigmoid activation function

        Args:
            z (float): XW + b

        Returns:
            float: sigmoid function
        """
        return 1/(1 + np.exp(-z))

    # Produce a neural network randomly initialized
    def initialize_parameters(self, n_x, n_h, n_y):
        """ 
        Randomly intialize parameters of the neural network

        Args:
            n_x (int): number of neurons in the first layer (input)
            n_h (int]): number of neurons in the hidden layer
            n_y (type): number of neuron in the last layer (output)

        Returns:
            sets: created parameters 
        """
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
    def forward_prop(self, X, parameters):
        """Forward propagation of the neural network

        Args:
            X (list): matrix of inputs
            parameters (sets): set of parameters

        Returns:
            A2 (list): Activation value for second layer
            cache(sets): includes Activation values of layer 1 and 2
        """
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
        A2 = self.sigmoid(Z2)

        cache = {
            "A1": A1,
            "A2": A2
        }
        return A2, cache

 
    def calculate_cost(self, A2, Y):
        """
        Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y
        We use the Mean Square Error cost function

        Args:
            A2 (list):  Activation value for first layer
            Y (list): output label

        Returns:
            float: MSE
        """
        # m is the number of examples
        cost = mean_squared_error(A2, Y)
        #cost = np.sum((0.5 * (A2 - Y) ** 2).mean(axis=1))/ m
        return cost

    # Apply the backpropagation
    def backward_prop(self, X, Y, cache, parameters):

        m = X.shape[1]

        A1 = cache["A1"]
        A2 = cache["A2"]

        W2 = parameters["W2"]

        # Compute the difference between the predicted value and the real values
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T)/ m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/ m
        # Because d/dx tanh(x) = 1 - tanh^2(x)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T)/ m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/ m

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return grads

    # Third phase of the learning algorithm: update the weights and bias
    def update_parameters(self, parameters, grads, learning_rate):
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
    def model(self, X, Y):
        parameters = self.initialize_parameters(self.n_x, self.n_h, self.n_y)
        for i in range(0, self.number_of_iterations + 1):
            a2, cache = self.forward_prop(X, parameters)
            cost = self.calculate_cost(a2, Y)
            grads = self.backward_prop(X, Y, cache, parameters)
            parameters = self.update_parameters(parameters, grads, self.learning_rate)
            if(i%100 == 0):
                print('Cost after iteration# {:d}: {:f}'.format(i, cost))

        return parameters

    # Make a prediction
    # X: represents the inputs
    # parameters: represents a model
    # the result is the prediction
    def predict(self, X, parameters):
        a2, cache = self.forward_prop(X, parameters)
        yhat = a2
        yhat = np.squeeze(yhat)
        if(yhat >= 0.5):
            y_predict = 1
        else:
            y_predict = 0

        return y_predict




