class Neuron:
    def __init__(self, inbound_neurons = []):
        # Properties will go here

        # Neurons from which this neuron receives values
        self.inbound_neurons  = inbound_neurons

        # Neurons to which this neuron passes values
        self.outbound_neurons = []

        # for each inbound neuron --> add this neuron as an outbound neuron
        for n in self.inbound_neurons :
            n.outbound_neurons.append(self)

        # This Neuron's calculated value
        self.value = None

        def forward(self):
            """
            Forward Propagation

            Compute the output value based on inbound_nodes and store the result in self.value

            :param self:
            :return:
            """
            raise NotImplemented

        def backward(self):
            """
            Backward Propagation

            :param self:
            :return:
            """
            raise NotImplemented