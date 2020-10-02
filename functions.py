import numpy as np

def softmax(x: list) -> list:
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Error asociado a un batch
def cost(pred: list, actual: list) -> float:
    return np.square(np.subtract(actual,pred)).mean(axis=0)

def relu_map(x):
    return 1 if x > 0 else 0

def sigmoid(z):
	return 1/(1 + np.exp(-z))

class Relu():
    """ Abstract the concept of ReLU function """

    def f(self, x: list) -> list:
        """ Compute function, where output is max(value, 0) """

        return np.array(list(map(np.max(), x)))

    def df(self, x: list) -> list:
        """
           
        """

        return np.array(list(map(relu_map(), x)))

class Sigmoid():

    # Define sigmoid
    def f(self, x: list) -> list:
        return np.array(list(map(sigmoid(), x)))
    
    # df sigmoid(x) = sigmoid(x)(1 - sigmoid(x))
    def df(self, x: list) -> list:
        return self.f(x)(1 - self.f(x))

