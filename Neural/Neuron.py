from .Common import *


class Neuron:

    def __init__(self, n: int) -> None:
        self.weights = []
        for _ in range(0, n):
            self.weights.append(1.0)
            self.bias = 1.0

    def edict(self, ins: list) -> float:
        out = 0.0
        for i in range(len(self.weights)):
            out = out + self.weights[i] * ins[i]
        out += self.bias * 1.0
        return sigmoid(out)

    def fit(self, ins: list, error: float) -> list:
        out = 0.0
        for i in range(len(self.weights)):
            out += self.weights[i] * ins[i]
        out += self.bias * 1.0
        out = sigmoid_derivate(out)
        #
        responsability = error * out
        gradients = []
        for i in range(len(self.weights)):
            dw = responsability * ins[i]
            gradients.append(dw)
            self.weights[i] += (dw * 0.1)
        self.bias += responsability
        return gradients

    def train(self, ins: list, expected: float):
        error = expected - self.edict(ins)
        self.fit(ins, error)

    def print(self) -> None:
        print("{}:{}".format(self.weights, self.bias))
