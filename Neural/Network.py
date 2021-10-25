from .Layer import *

class Network:

    def __init__(self, layout):
        self.layers = []
        for i in range(1, len(layout)):
            l = Layer(layout[i], layout[i - 1])
            self.layers.append(l)

    def edict(self, activations):
        for l in self.layers:
            activations = l.edict(activations)
        return activations

    def details(self, activations):
        details = []
        for l in self.layers:
            details.append(activations)
            activations = l.edict(activations)
        details.append(activations)
        return details

    def train(self, activations, derrors):
        details = self.details(activations)
        details.reverse()
        layers = list(self.layers)
        layers.reverse()
        for i in range(len(layers)):
            derrors = layers[i].train(details[i + 1], derrors)

    def print(self):
        for i in range(len(self.layers)):
            print("--- Layer {} ---".format(i))
            self.layers[i].print()