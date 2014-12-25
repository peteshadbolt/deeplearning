import numpy as np

class neuron:
    """ A sigmoid neuron """
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = np.random.uniform(0, 1, len(inputs))
        self.bias = np.random.rand()

    def get_output(self):
        input_values=[neuron.get_output() for neuron in self.inputs]
        dotproduct = sum(w*x for w,x in zip(self.weights, input_values))
        z = dotproduct + self.bias
        return 1/(1+np.exp(-z))

class source:
    """ A source - the first layer of the net """
    def __init__(self, value):
        self.value=value

    def get_output(self):
        return self.value

sources = [source(0) for i in range(10)]
first_layer = [neuron(sources) for i in range(10)]
second_layer = [neuron(first_layer) for i in range(10)]
output_node = neuron(second_layer)

print output_node.get_output()

