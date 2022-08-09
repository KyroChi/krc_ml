import numpy as np
from neural_net import *
from optimizers import SGD

def test1(p, f, seed: int=None):
    """
    Verifies that backpropogation correctly computes the derivatives of
    a small network.
    """
    if seed is not None:
        np.random.seed(seed)

    nn = NeuralNetwork(1)
    nn.add_layer(2, 'tanh', bias=False)
    nn.add_layer(1, 'tanh', bias=False)

    optimizer = SGD(learning_rate=0.00)
    nn.compile('mse', optimizer)

    i = 1

    o, a, ld = nn._feed_forward(np.array([1]), training=True)
    l = nn.loss(o, [i], differentiate=True)
    d = nn._back_prop(l, a, ld)

    w11 = nn.layers[0]['weights'][0][0]
    w12 = nn.layers[0]['weights'][0][1]
    w21 = nn.layers[1]['weights'][0][0]
    w22 = nn.layers[1]['weights'][1][0]
    outp = l * (1 - np.tanh(w21 * np.tanh(w11 * 1) + w22 * np.tanh(w12 * 1))**2)

    D = [
            np.array([[
                (outp * w21 * (1 - np.tanh(w11 * i)**2))[0][0],
                (outp * w22 * (1 - np.tanh(w12 * i)**2))[0][0]
            ]], dtype=np.float32),
            np.array([
                [outp * np.tanh(w11 * i)][0][0],
                [outp * np.tanh(w12 * i)][0][0]
            ], dtype=np.float32)
        ]

    D = np.array(D, dtype=object)
    d = np.array(d, dtype=object)

    error = np.sum(np.sum(D - d))

    if error < 1e-06:
        # TODO: Figure out if it is too much to require this to be 0
        p = p + 1
    else:
        f = f + 1

    return p, f

TESTS = [test1]

def run_tests(tests=TESTS, seed: int=None):
    if seed is not None:
        np.random.seed(seed)

    p = 0
    f = 0

    for test in tests:
        p, f = test(p, f, seed)

    print('passed: %d' % p)
    print('failed: %d' % f)
    print('pass rate: %.2f' % (p / (p + f)))

if __name__ == '__main__':
    run_tests()
