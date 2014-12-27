import numpy as np
import random

# TODO: I think that this code can be improved by going to 3D arrays

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
sigmoid_vec=np.vectorize(sigmoid)
def sigmoid_prime(z): return sigmoid(z)*(1-sigmoid(z))
sigmoid_prime_vec=np.vectorize(sigmoid_prime)

class NeuralNet:
    """ A neural net of sigmoid neurons """
    def __init__(self, widths):
        self.widths = widths
        self.num_layers = len(widths)
        self.weights = [np.random.randn(y, x) for x,y in zip(widths[:-1], widths[1:])]
        self.biases = [np.random.randn(y, 1) for y in widths[1:]]

    def feedforward(self, state):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            state = sigmoid(np.dot(w, state)+b)
        return state

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train the network against some training data. Eta is the learning rate"""
        n=len(training_data)
        for epoch in xrange(epochs):
            print "epoch %d" % epoch
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for index, mini_batch in enumerate(mini_batches):
                if (index % 1000)==0: print "minibatch %d" % index
                self.update_mini_batch(mini_batch, eta)
            if test_data: print "Won %d of %d test cases" % (self.evaluate(test_data), len(test_data))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by gradient descent using backpropagation """
        nabla_b = [np.zeros(b.shape) for b in self.biases] # A real number, the gradient, for each bias in the network
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # A real number, the gradient, for each weight in the network
        # This takes an average over many minibatches, finding nabla_b and nabla_w - the direction in which to move
        for x, y in mini_batch:
            # x is the image data, y is the label
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # This gets the gradient via backpropagation magic
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Now move in the opposite direction to the gradient, at a rate eta
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Get the gradient of the cost function in bias and weight DOFs, given an image and a label """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # TODO: This seems to duplicate existing functionality in self.feedforward
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result. """
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations."""
        return (output_activations-y) 

    def __str__(self): return "Neural Net %s" % self.widths



if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    epochs = 5
    mini_batch_size = 5
    eta = 0.1 

    net=NeuralNet([28*28, 15, 10])
    net.train(training_data, epochs, mini_batch_size, eta, test_data=test_data)
