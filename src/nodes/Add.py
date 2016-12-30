from .Neuron import Neuron

class Add(Neuron):
    def __init__(self, x, y):
        Neuron.__init__(self, [x, y])

    def forward(self):
        """
        Adds the values of all incoming nodes together and sets the sum to current node's value

        :return:
        """
        self.value = 0
        for n in self.inbound_neurons:
            self.value += n.value