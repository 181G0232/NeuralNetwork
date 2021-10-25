from .Neuron import *

class Layer:

    def __init__(self, nc, wc):
        self.neurons = []
        for _ in range(nc):
            neuron = Neuron(wc)
            self.neurons.append(neuron)

    def edict(self, activations):
        edicts = []
        for n in self.neurons:
            e = n.edict(activations)
            edicts.append(e)
        return edicts

    def train(self, activations, derrors):
        berrors = []
        for _ in range(len(activations)):
            berrors.append(0.0)
        for i in range(len(self.neurons)):
            bes = self.neurons[i].train(activations, derrors[i])
            for j in range(len(bes)):
                berrors[j] += bes[j]
        return berrors

    def print(self):
        for n in self.neurons:
            n.print()