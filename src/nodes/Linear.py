from .Neuron import Neuron

class Linear(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other neurons.
        # The weight and bias values are stored within the
        # respective neurons.
        self.weights = weights
        self.bias = bias

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        self.value = self.bias.value
        for i in range(len(self.inbound_neurons)):
            self.value += self.inbound_neurons[i].value * self.weights[i].value