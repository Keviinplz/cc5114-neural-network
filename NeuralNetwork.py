import numpy as np

class NeuralNetwork():

    def __init__(self, n_i, l_h, l_a, n_y, number_of_epoch, learning_rate):
        self.n_i = n_i
        self.l_h = l_h
        self.l_a = l_a
        self.n_y = n_y 
        self.w_l = []
        self.b_l = []
        self.number_of_epoch
        self.learning_rate

    """
    Inicializar todos los parámetros
    n_x cantidad de neuronas de entrada
    m_y cantidad de neuronas de salidad
    i_h cantidad de capas escondida, donde cada capa escondida tiene a su vez k neuronas

    Con toda esta información, vamos a crear matrices con números aleatorios
    estos serán nuestras matrices de peso (W)

    Haremos lo mismo con los bias


    Por ejemplo:

    input = dim 6
    hidd  = [dim 50, dim 30]
    out   = dim 18

    W_0 dim(6, 50)
    W_1 dim(50, 30)
    W_2 dim(30, 18)
    """
    def initialize(self):
        # Inicialización de los Pesos
        inputLayer = np.random.randn(self.n_i, self.l_h[0])
        self.w_l.append(inputLayer)

        for i in range(len(self.l_h) - 1):
            param = np.random.randn(self.l_h[i], self.l_h[i+1])
            self.w_l.append(param)

        outputLayer = np.random.rand(self.l_h[len(self.l_h) - 1], self.n_y)
        self.w_l.append(outputLayer)

        # Inicialización de los Bias
        for j in range(len(self.l_h)):
            bias = np.zeros(l_h[j])
            self.b_l.append(bias)

        self.b_l.append(np.zeros(n_y))

    """
    X
    h1
    y

    h1 =f(x*w1 + b1)
    h2 =f(h1*w2 + b2)
    ...
    h5 = f(h4*w5 + b5)
    y = f(h5*w6 + b6)
    y_predicted = softmax(y)
    """


    def forward(self, x):
        lista = []
        u = np.dot(x, self.w_l[0]) + self.b_l[0]
        h = self.l_a[0](u)
        lista.append(h)

        for i in range(len(self.w_l) - 2):
            # Hago la iteración por la capa iesima 
            ui = np.dot(lista[i], self.w_l[i+1]) + self.b_l[i+1]
            hi = self.l_a[i+1](ui)                              
            lista.append(hi)

        # Obligo a la red a tener más de 1 capa 
        u_f = np.dot(lista[len(self.w_l) - 2], self.w_l[len(self.w_l) - 1]) + self.b_l[len(self.w_l) - 1]
        y_predicted = softmax(u_f)
        return y_predicted




