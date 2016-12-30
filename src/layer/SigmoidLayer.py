from .Layer import Layer
import numpy as np

class Sigmoid(Layer):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        # 1 / (1 + e ^ (-x))
        return 1. / (1. + np.exp(-x))



    def forward(self):
        """
        Set the value of this layer to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        self.value = self._sigmoid(self.inbound_layers[0].value)