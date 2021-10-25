from Neural import *

network = Network([2, 4, 3])

expecteds = [
    [[0, 0], [0, 0, 0]],
    [[0, 1], [1, 0, 1]],
    [[1, 0], [1, 0, 1]],
    [[1, 1], [1, 1, 0]]
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
        print("{} -> [".format(e), end="")
        print("%.2f" % outs[0], end="")
        for i in range(1, len(outs)):
            print(", %.2f" % outs[i], end="")
        print("]")

while True:
    option = input("train or test? : ")
    if option == "print":
        network.print()
    elif option == "train":
        train(1000)
        test()
    elif option == "test":
        x = float(input("x: "))
        y = float(input("y: "))
        rs = network.edict([x, y])
        print("[%.2f" % rs[0], end="")
        for i in range(1, len(rs)):
            print(", %.2f" % rs[i], end="")
        print("]")
    elif option == 'exit':
        break