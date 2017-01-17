import numpy as np
import math

def sigmoid(num, deriv):
    if not deriv:
        return 1 / (float) (1 + np.exp(-1 * num))
    else:
        return num * (1 - num)

#leaky ReLU
def relu(num, deriv):
    if not deriv:
        return max(num, 0.01 * num)
    else:
        return int(num > 0)

def softmax(nums):
    total = sum([int(np.exp(num)) for num in nums])
    return [100 * np.exp(num) / float(total) for num in nums]

class Neuron(object):

    def __init__(self, features, bias = False, funct = 'sigmoid'):
        self.features = features
        self.bias = int(bias)
        self.funct = sigmoid if funct is 'sigmoid' else relu
        self.weights = np.random.random(features) - 0.5
        self.sample = None
        self.activation = 0

    """
    Given a sample, output the neuron's activation.
    Activation may be calculated using sigmoid or ReLU functions.
    """
    def activate(self, sample):
        dot = np.dot(self.weights, sample) + self.bias
        self.sample = np.array(sample)
        self.activation = self.funct(dot, False)
        return self.activation

    """
    Used to update weights through stochastic gradient descent.
    Gradients are calculated using the derivative of the activation
    function, and the neuron's activation.
    """
    def gradient(self):
        return self.funct(self.activation, True)

    def train(self, delta, alpha):
        self.bias += delta * alpha
        self.weights += delta * alpha * self.sample

class Layer(object):

    def __init__(self, neurons, features, alpha, bias = True):
        self.funct = 'sigmoid'
        self.neurons = [Neuron(features, bias, self.funct) for i in range(neurons)]
        self.alpha = alpha
        self.bias = bias

    """
    Forward propogates an input through the neuron layer, and
    returns inputs for the following layer.
    """
    def forward(self, sample):
        output = []
        for neuron in self.neurons:
            output += [neuron.activate(sample)]
        return np.array(output)

    """
    Performs backpropogation of error through the layer, and
    uses stochastic gradient descent to update neuron weights.
    """
    def backprop(self, error):
        new_error = np.zeros(self.neurons[0].features)
        for neuron in self.neurons:
            delta = neuron.gradient() * error.pop(0)
            new_error += delta * np.array(neuron.weights)
            neuron.train(delta, self.alpha)
        return new_error

class Network(object):

    def __init__(self, layers, alpha = 0.01, bias = True):
        self.structure = zip(layers[:-1], layers[1:])
        self.network = []
        self.features = layers[0]
        self.labels = layers[-1]
        self.alpha = alpha
        self.bias = bias

        #builds a network based on features, layers, and labels
        for struct in self.structure:
            self.network.append(Layer(struct[1], struct[0], alpha, bias))

    """
    Forward propogates a sample through an entire network
    """
    def forward(self, sample):
        for layer in self.network:
            sample = layer.forward(sample)
        return sample

    """
    Backpropogates an error vector through each layer of the
    network and performs learning updates on weight vectors.
    """
    def backprop(self, error):
        self.network.reverse()
        for layer in self.network:
            error = layer.backprop(error.tolist())
        self.network.reverse()

    """
    From a sample and training label, calculates the error and
    performs backpropogation and weight updates on all neurons
    in the network.
    """
    def train(self, sample, label):
        output = self.forward(sample)
        error = np.array(label) - np.array(output)
        self.backprop(error)

    def classify(self, sample):
        output = self.forward(sample).tolist()
        return output.index(max(output))
