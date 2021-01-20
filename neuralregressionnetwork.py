import numpy as np

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def get_activation_function(activation_function, alpha):
    if activation_function == "alt_sigmoid":
        act_func = lambda a: a / (1 + np.abs(a))
        act_func_der = lambda a: 1 / (1 + np.abs(a)) ** 2
        return act_func, act_func_der
    elif activation_function == "threshold":
        act_func = lambda arr: np.where(arr > 0, 1, 0)
        act_func_der = lambda a: 0
        return act_func, act_func_der
    elif activation_function == "tanh":
        act_func = lambda a: np.tanh(a)
        act_func_der = lambda a: 1 - np.tanh(a) ** 2
        return act_func, act_func_der
    elif activation_function == "ReLU":
        act_func = lambda arr: arr * np.where(arr >= 0, 1, 0)
        act_func_der = lambda arr: np.where(arr >= 0, 1, 0)
        return act_func, act_func_der
    elif activation_function == "logistic":
        act_func = lambda a: 1 / (1 + np.exp(-a))
        act_func_der = lambda a: act_func(a) * (1 - act_func(a))
        return act_func, act_func_der
    elif activation_function == "ELU":
        act_func = lambda arr: np.where(arr >= 0, arr, alpha * (np.exp(arr) - 1))
        act_func_der = lambda a: np.where(a >= 0, 1, alpha * np.exp(a))
        return act_func, act_func_der
    else:
        raise NameError('activation function is unknown')


def add_1_column(m: np.array):
    x = np.ones((m.shape[0], m.shape[1]))
    x[:, :-1] = m
    return x


# implementation of a neural network that uses MSE for regression.
# network has variable hidden layers with variable amounts of
# neurons that are non-linear and one linear output neuron
class NeuralRegressionNetwork:

    def __init__(self, no_of_layers=1, no_of_neurons: list = None, activation_function="alt_sigmoid", alpha=1):
        if no_of_neurons is None:
            no_of_neurons = []
            for i in range(no_of_layers):
                no_of_neurons.append(5)
        self.no_of_neurons = no_of_neurons
        self.weight_vectors = []
        self.bias_vectors = []
        self.act_func, self.act_func_der = get_activation_function(activation_function, alpha)

    def initialize_params(self, x_nodes):
        rand = lambda x, y: np.random.randn(x, y)
        self.weight_vectors.append(rand(self.no_of_neurons[0], x_nodes))
        self.bias_vectors.append(rand(self.no_of_neurons[0], 1))
        for i in range(1, len(self.no_of_neurons)):
            self.weight_vectors.append(rand(self.no_of_neurons[i], self.no_of_neurons[i - 1]))
            self.bias_vectors.append(rand(self.no_of_neurons[i], 1))
        self.weight_vectors.append(rand(1, self.no_of_neurons[len(self.no_of_neurons) - 1]))
        self.bias_vectors.append(rand(1, 1))

    def feedForward(self, x):
        z_s = []
        a_s = []
        for i in range(len(self.weight_vectors)):
            if i == 0:
                z_s.append(np.dot(self.weight_vectors[0], x) + self.bias_vectors[0])
                a_s.append(self.act_func(z_s[0]))
            else:
                z_s.append(np.dot(self.weight_vectors[i], a_s[i - 1]))
                if i == len(self.weight_vectors) - 1:
                    a_s.append(z_s[i])
                else:
                    a_s.append(self.act_func(z_s[i]))
        return z_s, a_s

    def backwardPropagate(self, x, y, a_s):
        m = len(y)
        grads = []
        b_grads = []
        for i in range(len(self.weight_vectors) - 1, -1, -1):
            if i == len(self.weight_vectors) - 1:
                dA = 1 / m * (a_s[i] - y)
                dZ = dA
            else:
                dA = np.dot(self.weight_vectors[i + 1].T, dZ)
                dZ = np.multiply(dA, self.act_func_der(a_s[i]))
            if i == 0:
                grads.append(1 / m * np.dot(dZ, x.T))
                b_grads.append(1 / m * np.sum(dZ, axis=1, keepdims=True))
            else:
                grads.append(1 / m * np.dot(dZ, a_s[i - 1].T))
                b_grads.append(1 / m * np.sum(dZ, axis=1, keepdims=True))
        grads.reverse()
        b_grads.reverse()
        return grads, b_grads

    def update_params(self, grads, b_grads, learning_rate):
        for i in range(len(self.weight_vectors)):
            self.weight_vectors[i] -= learning_rate * grads[i]
            self.bias_vectors[i] -= learning_rate * b_grads[i]

    def train(self, x, y, epochs=100, learning_rate=0.1, x_val=np.array([]), y_val=np.array([])):
        ls = []
        self.initialize_params(x.shape[0])
        if epochs == "auto":
            es = []
            gs = []
            while len(ls) < 100000:
                z_s, a_s = self.feedForward(x)
                ls.append(mean_squared_error(y, a_s[len(a_s) - 1]))

                _, ae_s = self.feedForward(x_val)
                es.append(mean_squared_error(y_val, ae_s[len(ae_s) - 1]))

                grads, b_grads = self.backwardPropagate(x, y, a_s)

                gs.append(norm(grads[0], b_grads[0]))
                #if len(ls) > 2 and ls[len(ls) - 1] > ls[len(ls) - 2]:
                #    break
                self.update_params(grads, b_grads, learning_rate)
            return ls, es, gs
        else:
            fig, ax = plt.subplots(dpi=150)
            x_test = range(-1500, 1500, 5)
            x_test = np.array(x_test) / 100
            y_real = np.sinc(x_test/np.pi)
            ax.plot(x_test, y_real, label='sinc')
            for i in range(epochs):
                z_s, a_s = self.feedForward(x)
                ls.append(mean_squared_error(y, a_s[len(a_s) - 1]))
                # if i % 1000 == 0:
                #     x_p = np.reshape(x_test, (-1, 1)).T
                #     _, a_es = self.feedForward(x_p)
                #     y_pred = np.reshape(a_es[len(a_s) - 1], -1)
                #     ax.plot(x_test, y_pred, label='i=' + str(i))
                grads, b_grads = self.backwardPropagate(x, y, a_s)
                self.update_params(grads, b_grads, learning_rate)
            ax.legend()
            return ls


def norm(x, y):
    return np.sqrt(np.sum(x ** 2,axis=0) + y ** 2)[0]