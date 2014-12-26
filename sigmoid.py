import numpy as np
import pickle
import random

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

class NeuralNet:
    """ A neural net of sigmoid neurons """
    def __init__(self, widths):
        self.widths = widths
        self.weights = [np.random.randn(y, x) for x,y in zip(widths[:-1], widths[1:])]
        self.biases = [np.random.randn(y) for y in widths[1:]]

    def feedforward(self, state):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            state = sigmoid(np.dot(w, state)+b)
        return state

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train the network against some training data. Eta is the learning rate"""
        n=len(training_data)
        for epoch in xrange(epochs):
            training_data=random.shuffle(training_data)
            mini_batches = training_data[k:k+mini_batch_size for k in range(0, n mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data: self.test(test_data)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by gradient descent using backpropagation """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
            

    def test(self, test_data):
        """ Test the network against some test data """
        pass

    def __str__(self): return "Neural Net %s" % self.widths


def load_data():
    f=open("nielsen/data/mnist.pkl", "r")
    data=pickle.load(f)
    f.close()
    return data


if __name__ == '__main__':
    training_data=[]
    epochs = 5
    mini_batch_size = 5
    eta = 0.1 # Learning rate

    n=NeuralNet([28*28, 15, 10])

    output = n.feedforward(np.zeros(28*28))
    print output

    #n.train([], epochs, mini_batch_size, eta)



