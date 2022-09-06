import numpy as np

class Initializer(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_shape, output_shape):
        raise NotImplementedError(
            "Must implement 'Initializer' primiative"
        )

class GlorotInitializer(Initializer):
    def __init__(self):
        super().__init__('Glorot')

    def __call__(self, input_shape, output_shape):
        sigma2 = 2 / ( input_shape + output_shape )
        return np.random.normal(
            loc=0, scale=sigma2, size=(input_shape, output_shape)
        )

def get_initializer(activation: str):
    if activation == 'glorot':
        return GlorotInitializer()
    else:
        raise Exception(
            "Unrecognized initializer %s" % activation
        )
