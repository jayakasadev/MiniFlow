"""
Test your MSE method with this script!

No changes necessary, but feel free to play
with this script to test your network.
"""

import numpy as np
from src.layer.InputLayer import Input
from src.layer.MSELayer import MSE
from src.util.sort.TopologicalSortLayers import  topological_sort
from src.util.forwardPass.ForwardPass import forward_pass

y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(graph)

"""
Expected output

23.4166666667
"""
print(cost.value)
