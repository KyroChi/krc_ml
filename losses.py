import numpy as np

class Loss(object):
    def __init__(self, name, func, deriv):
        self.name = name
        self.function = func
        self.derivative = deriv

    def __call__(self, y, yhat, differentiate=False):
        if differentiate:
            return self.derivative(y - yhat)
        else:
            return self.function(y - yhat)

MSE = Loss(
    'mse',
    lambda x: np.mean(x**2),
    lambda x: 2*x
)

def get_loss(loss: str):
    if loss == 'mse':
        return MSE
    else:
        raise Exception('Unrecognized loss %s' % activation)
