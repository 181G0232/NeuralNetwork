import math

def cost(activation, expect):
    tmp = ((expect - activation) ** 2.0) / 2.0
    return tmp

# sigmoide
def activation(signal):
    # tmp = 1.0 / (1.0 + math.exp(-signal))
    tmp = math.tanh(signal)
    return tmp

# sigmoide derivate
def dactivation(signal):
    # tmp = activation(signal) * (1.0 - activation(signal))
    tmp = 1.0 - (activation(signal) ** 2)
    return tmp

def derror(signal, activation, expect):
    tmp = (expect - activation) * dactivation(signal)

# update last derror: derror(signale, activation, expect)
# update last gradient: signal * derror

# update last-1 derror: weight * dactivation(signal) * derror
# update last-1 gradient: signal * derror

 