from .Layer import Layer
import numpy as np

class Sigmoid(Layer):
    """
    Represents a layer that performs the sigmoid activation function.
    """
    def __init__(self, layer):
        # The base class constructor.
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            """
            TODO: Your code goes here!
            
            Set the gradients property to the gradients with respect to each input.
            
            NOTE: See the Linear layer and MSE layer for examples.

            this method sums the derivative (it's a normal derivative when there;s only one variable) with respect to
            the only input over all the output layers

            ​​(∂sigmoid​​​ / ​∂x) * (∂cost​​ / ∂sigmoid​​)

            (∂sigmoid​​​ / ​∂x) = sigmoid * (1 - sigmoid)

            (∂cost​​ / ∂sigmoid​​) = grad_cost
            """
            sigmoid = self.value

            # for each input value in X, calculate the corresponding gradient
            self.gradients[self.inbound_layers[0]] += sigmoid * (1 - sigmoid) * grad_cost