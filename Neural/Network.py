from .Layer import *

class Network:
    
    def __init__(self, layout:list) -> None:
        self.layers = []
        for i in range(1, len(layout)):
            self.layers.append(Layer(layout[i], layout[i - 1]))

    def edict(self, ins:list) -> list:
        for l in self.layers:
            ins = l.edict(ins)
        return ins

    def edictDetails(self, ins:list) -> list:
        details = []
        for l in self.layers:
            details.append(ins)
            ins = l.edict(ins)
        return details

    def fit(self, ins:list, errors:list) -> None:
        details = self.edictDetails(ins)
        details.reverse()
        details.append(ins)
        layers = list(self.layers)
        layers.reverse()
        for i in range(len(layers)):
            errors = layers[i].fit(details[i], errors)

    def train(self, ins:list, expecteds:list) -> None:
        edicts = self.edict(ins)
        errors = []
        for i in range(len(expecteds)):
            errors.append(expecteds[i] - edicts[i])
        self.fit(ins, errors)

    def print(self) -> None:
        for l in self.layers:
            l.print()
        
