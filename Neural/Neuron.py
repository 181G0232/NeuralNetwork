import math
import random
from .Common import *

class Neuron:

    def __init__(self, n):
        self.weights = []
        for _ in range(n):
            self.weights.append(float(random.randint(-1, 1)))
        self.bias = random.randint(-1, 1)

    def ponderate(self, activations):
        ponderation = 0.0
        for i in range(len(self.weights)):
            ponderation += activations[i] * self.weights[i]
        ponderation += self.bias
        return ponderation

    def edict(self, activations):
        return activation(self.ponderate(activations))

    def train(self, activations, derror):
        signal = self.ponderate(activations)
        derror *= dactivation(signal)
        berrors = []
        for i in range(len(self.weights)):
            self.weights[i] += derror * activations[i] * 0.1
            be = derror * self.weights[i]
            berrors.append(be)
        self.bias += derror * 0.1
        return berrors

    def print(self):
        print("[", end="")
        for w in self.weights:
            print("%.2f, " % w, end="")
        print("%.2f]" % self.bias)