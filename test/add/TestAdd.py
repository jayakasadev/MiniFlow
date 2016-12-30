"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from src.nodes.Add import Add
from src.nodes.Input import Input
from src.util.sort.TopologicalSortNeurons import topological_sort
from src.util.forwardPass.ForwardPass import forward_pass

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)
# output = None

# NOTE: because topological_sort set the values for the `Input` neurons we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))