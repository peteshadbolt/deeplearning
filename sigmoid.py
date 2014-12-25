import numpy as np

class NeuralNet:
    """ A neural net of sigmoid neurons """
    def __init__(self, widths):
        self.widths = widths
        self.nlayers = len(widths)
        shapes = zip(widths[:-1], widths[1:])
        self.weights = [np.random.randn(y, x) for x,y in shapes]
        self.biases = [np.random.randn(y,1) for y in widths[1:]]

    def __str__(self): return "Neural Net %s" % self.widths

n=NeuralNet([28*28, 15, 10])
print n


