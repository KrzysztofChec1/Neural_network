import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Definicja klasy Neuron (już dostarczona)
class Neuron:
    def __init__(self, n_inputs, bias = 0., weights = None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x): #activation function (here: leaky_relu)
        return max(x*.1, x)

    def __call__(self, xs): #calculate the neuron's output: multiply the inputs with the weights and sum the values together, add the bias value,
                            # then transform the value via an activation function
        return self._f(xs @ self.ws + self.b) #Mnozenie macierzy @

# Definicja klasy dla warstwy sieci neuronowej
class NeuralLayer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)] # towrzy warstwe z n nuronys

    def __call__(self, xs):
        return np.array([neuron(xs) for neuron in self.neurons]) #przechodzimy przez wszystkie nuerony

class NeuralNetwork:
    def __init__(self, structure):
        self.layers = []
        for i in range(len(structure) - 1):
            self.layers.append(NeuralLayer(structure[i], structure[i+1]))

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


network_structure = [3, 4, 4, 1]
network = NeuralNetwork(network_structure)

def visualize_network(structure):
    G = nx.DiGraph()
    subset_keys = {}  # Słownik przechowujący przypisania węzłów do warstw
    for i, layer_size in enumerate(structure):
        for j in range(layer_size):
            neuron_name = f'Layer {i + 1} - Neuron {j + 1}'
            G.add_node(neuron_name)
            subset_keys[neuron_name] = i  # Przypisanie węzła do warstwy
        if i > 0:
            previous_layer_size = structure[i - 1]
            for prev_j in range(previous_layer_size):
                for curr_j in range(layer_size):
                    prev_neuron = f'Layer {i} - Neuron {prev_j + 1}'
                    curr_neuron = f'Layer {i + 1} - Neuron {curr_j + 1}'
                    G.add_edge(prev_neuron, curr_neuron)

    # Przygotowanie struktury subset_keys do użycia w multipartite_layout
    subsets = {}
    for node, subset in subset_keys.items():
        if subset not in subsets:
            subsets[subset] = []
        subsets[subset].append(node)

    pos = nx.multipartite_layout(G, subset_key=subsets)  # Poprawne użycie subset_key
    nx.draw(G, pos, with_labels=True)
    plt.show()


# Wizualizacja sieci
visualize_network(network_structure)
