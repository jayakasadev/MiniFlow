from .Layer import Layer
import numpy as np

# mean squared error
class MSE(Layer):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last layer for a network.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_layers[0].value
        a = self.inbound_layers[1].value
        self.value = np.mean(np.square(y-a))