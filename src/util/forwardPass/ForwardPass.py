def forward_pass(output, sorted):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_neuron`: A neuron in the graph, should be the output neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output neuron's value
    """

    for n in sorted:
        n.forward()

    return output.value

def forward_pass(graph):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()
