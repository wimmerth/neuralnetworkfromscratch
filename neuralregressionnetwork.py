import numpy as np
from sklearn.metrics import mean_squared_error


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


def transform(x):
    if x.ndim == 1:
        return np.reshape(x, (-1, 1))
    else:
        return x


def add_bias(x):
    y = np.ones((x.shape[0], x.shape[1] + 1))
    y[:, :-1] = x
    return y


def remove_bias(x):
    return x[:, :-1]


def norm(x):
    return np.mean(np.sqrt(np.sum(x ** 2, axis=0)))


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
        self.weights = []
        self.act_func, self.act_func_der = get_activation_function(activation_function, alpha)

    def initialize_params(self, x_nodes):
        rand = lambda x, y: np.random.randn(x, y)
        # first layer
        self.weights.append(rand(x_nodes + 1, self.no_of_neurons[0]))
        # mid layers
        for i in range(1, len(self.no_of_neurons)):
            self.weights.append(rand(self.no_of_neurons[i - 1] + 1, self.no_of_neurons[i]))
        # last layer
        self.weights.append(rand(self.no_of_neurons[len(self.no_of_neurons) - 1] + 1, 1))

    # using last column of feedForward as prediction
    def predict(self, x):
        x = transform(x)
        a_s = self.feedForward(x)
        return a_s[len(a_s) - 1]

    # measure score as MSE
    def score(self, x, y):
        x = transform(x)
        y = transform(y)
        a_s = self.feedForward(x)
        return mean_squared_error(y, a_s[len(a_s) - 1])

    # feed data through network values for neutrons + transformed values after applying activation function
    def feedForward(self, x):
        a_s = []
        for i in range(len(self.weights)):
            if i == 0:
                a_s.append(self.act_func(np.dot(add_bias(x), self.weights[0])))
            elif i == len(self.weights) - 1:
                a_s.append(np.dot(add_bias(a_s[i - 1]), self.weights[i]))
            else:
                a_s.append(self.act_func(np.dot(add_bias(a_s[i - 1]), self.weights[i])))
        return a_s

    # perform backward propagation
    def backwardPropagate(self, x, y, a_s):
        m = len(y)
        deltas = []
        derivatives = []
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                # since we use linear output, no use of f'
                delta = (y - a_s[i]) / m
                deltas.append(delta)
            else:
                delta = remove_bias(np.dot(deltas[len(deltas) - 1], self.weights[i + 1].T)) * self.act_func_der(a_s[i])
                deltas.append(delta)
            if i == 0:
                derivatives.append(np.dot(delta.T, add_bias(x)))
            else:
                derivatives.append(np.dot(delta.T, add_bias(a_s[i - 1])))
        derivatives.reverse()
        return derivatives

    # update weights and bias with precomputed gradients
    def update_params(self, grads, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * grads[i].T

    def train(self, x, y, epochs=100, learning_rate=0.1, x_val=None, y_val=None):
        x = transform(x)
        y = transform(y)
        ls = []
        self.initialize_params(x.shape[1])
        # perform 50000 learning iterations and return tables with train- and test-errors as well as gradient norms
        if epochs == "test":
            if x_val is None or y_val is None:
                raise ValueError
            x_val = transform(x_val)
            y_val = transform(y_val)
            es = []
            gs = []
            for i in range(50000):
                a_s = self.feedForward(x)
                # compute MSE on train labels
                ls.append(mean_squared_error(y, a_s[len(a_s) - 1]))
                # feedForward can be used as prediction function
                ae_s = self.feedForward(x_val)
                # compute MSE on test labels
                es.append(mean_squared_error(y_val, ae_s[len(ae_s) - 1]))
                # compute gradients for bias and weights
                ders = self.backwardPropagate(x, y, a_s)
                # compute gradient norms
                n = sum([norm(ders[i]) for i in range(len(self.weights))])
                gs.append(n)
                # update weights and bias with computed gradients
                self.update_params(ders, learning_rate)
            return ls, es, gs
        else:
            for i in range(epochs):
                a_s = self.feedForward(x)
                # compute MSE on train labels
                ls.append(mean_squared_error(y, a_s[len(a_s) - 1]))
                # compute gradients for bias and weights
                ders = self.backwardPropagate(x, y, a_s)
                # update weights and bias with computed gradients
                self.update_params(ders, learning_rate)
            return ls
