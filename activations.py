import numpy as np

class Activation(object):
    def __init__(self, name, activ, deriv):
        self.name = name
        self.activation = activ
        self.derivative = deriv

    def __call__(self, X, differentiate=False):
        if differentiate:
            return self.derivative(X)
        else:
            return self.activation(X)

IDENTITY = Activation(
    'identity',
    lambda x: x,
    np.vectorize(lambda x: 0)
)
    
TANH = Activation(
    'tanh',
    np.vectorize(
        lambda x: np.tanh(x)
    ),
    np.vectorize(
        lambda x: 1 - np.tanh(x)**2
    )
)

SIGMOID = Activation(
    'sigmoid',
    np.vectorize(
        lambda x: np.exp(x) / (np.exp(x) + 1)
    ),
    np.vectorize(
        lambda x: np.exp(-x) / (np.exp(-x) + 1)**2
    )
)

RELU = Activation(
    'relu',
    np.vectorize(lambda x: x if x > 0 else 0),
    np.vectorize(lambda x: 1 if x > 0 else 0),
)

def get_activation(activation: str):
    if activation == 'tanh':
        return TANH
    elif activation == 'sigmoid':
        return SIGMOID
    elif activation == 'relu':
        return RELU
    elif activation == 'identity':
        return IDENTITY
    else:
        raise Exception('Unrecognized activation %s' % activation)
