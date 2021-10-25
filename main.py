from Neural import *

network = Network([2, 3, 2])

expecteds = [
    [[0, 0], [0, 1]],
    [[0, 1], [0, 1]],
    [[0, -1], [0, 1]],
    [[0.5, 1], [-1, 1]],
    [[0.5, -1], [1, 1]],
    [[0.5, 0], [0, 1]],
    [[1, 1], [0, -1]],
    [[1, -1], [0, -1]],
    [[1, 0], [0, -1]],
    [[-1, 0], [0, 1]],
    [[-1, -1], [0, 1]],
    [[-1, 1], [0, 1]]
]

def train(times):
    for _ in range(times):
        for e in expecteds:
            errors = []
            edicts = network.edict(e[0])
            for i in range(len(edicts)):
                errors.append(e[1][i] - edicts[i])
            network.train(e[0], errors)

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