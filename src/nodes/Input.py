# have to prefix the file name with the . operator
from .Neuron import Neuron

class Input(Neuron):
    def __init__(self):
        # An input neuron has no inbound nodes, so there is no need to pass anything to Neuron parent
        Neuron.__init__(self)

    """
    NOTE: Input neuron is the only node where the value may be passed as an argument to forward()

    All other neuron implementations should get the value of the previous neuron from self.inbound_nodes

    Example:
    val0 = self.inbound_nodes[0].value
    """
    def forward(self, value = None):
        # Overwrite the valye if one is passed in
        if value is not None:
            self.value = value
