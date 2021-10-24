from .Common import *
from .Neuron import *


class Layer:

    def __init__(self, nc: int, wc: int) -> None:
        self.neurons = []
        for i in range(nc):
            self.neurons.append(Neuron(wc))

    def print(self) -> None:
        for n in self.neurons:
            print(n)

    def edict(self, ins: list) -> list:
        outs = []
        for n in self.neurons:
            outs.append(n.edict(ins))
        return outs

    def fit(self, ins: list, errors: list) -> list:
        gradients = []
        for _ in range(len(ins)):
            gradients.append(0.0)
        #
        for i in range(len(errors)):
            grads = self.neurons[i].fit(ins, errors[i])
            for j in range(len(ins)):
                gradients[j] += grads[j]
        #
        return gradients

    def train(self, ins: list, expecteds: list) -> None:
        edicts = self.edict(ins)
        errors = []
        for i in range(len(expecteds)):
            errors.append(expecteds[i] - edicts[i])
        self.fit(ins, errors)

    def print(self) -> None:
        for n in self.neurons:
            n.print()
