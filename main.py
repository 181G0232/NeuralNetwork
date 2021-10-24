from Neural import *

network = Network([2, 2, 1])

expecteds = [
    [[0, 0], [0]],
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]]
]

def train(times:int):
    for _ in range(times):
        for e in expecteds:
            network.train(e[0], e[1])

def test():
    network.print()
    for e in expecteds:
        outs = network.edict(e[0])
        print("{} -> {}".format(e, outs))

while True:
    test()
    option = input("train? : ")
    if option == 'y':
        train(1000)
    elif option == 'n':
        break