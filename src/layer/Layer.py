class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError