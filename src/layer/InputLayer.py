from .Layer import Layer

class Input(Layer):
    """
    While it may be strange to consider an input a layer when
    an input is only an individual node in a layer, for the sake
    of simpler code we'll still use Layer as the base class.

    Think of Input as collating many individual input nodes into
    a Layer.
    """
    def __init__(self):
        # An Input layer has no inbound layers,
        # so no need to pass anything to the Layer instantiator
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input Layer has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_Layers:
            self.gradients[self] += n.gradients[self]