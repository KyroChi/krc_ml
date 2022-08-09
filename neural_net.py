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
        lambda x: -np.exp(-x) / (np.exp(-x) + 1)**2
    )
)

def get_activation(activation: str):
    if activation == 'tanh':
        return TANH
    elif activation == 'sigmoid':
        return SIGMOID
    elif activation == 'identity':
        return IDENTITY
    else:
        raise Exception('Unrecognized activation %s' % activation)

    
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


class Optimizer(object):
    def __init__(self, name):
        self.name = name

    def update(self, target, direction):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, target, direction):
        return target - self.lr * direction


def get_optimizer(optimizer: str):
    if optimizer == 'sgd':
        return SGD(learning_rate=0.1)
    else:
        raise Exception('Unrecognized optimizer %s' % optimizer)
    
class NeuralNetwork(object):
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size
        self.compiled = False

    def add_layer(
            self,
            width,
            activation,
            bias=True,
            initializer='he'
    ):
        """
        Adds a fully connected layer to the network. Can include bias.
        """

        if len(self.layers) == 0:
            prev_width = self.input_size
        else:
            prev_width = self.layers[-1]['weights'].shape[-1]
        
        weights = 2 * np.random.random_sample(
            size=(prev_width, width)
        ) - 1
        
        layer = {
            'activation': activation,
            # 'bias': 2 * np.random.random_sample(size=(1,)) - 1,
            'bias': 0,
            'weights': weights
        }
        
        self.layers.append(layer)
        return

    def _feed_forward(self, X, training=False):
        """
        X is input

        if training is False: returns the output from the neural 
        network.

        if training is True: returns the output and the 
        pre-activation values from the neural network.
        """
        if training:
            activations = [X]
            derivatives = []
            
        out = X
        for layer in self.layers:
            activation = get_activation(layer['activation'])
            weights = layer['weights']

            mat = out @ weights
            out = activation(mat.reshape(1, -1))

            if training:
                activations.append(out)
                derivatives.append(
                    activation(mat.reshape(1, -1), differentiate=True)
                )
                
        if training:
            return out, activations, derivatives
        else:
            return out

    def _back_prop(self, loss, activations, local_derivatives):
        # Walk through layers backwards, return a list of matricies
        # where the entries are the partial derivatives with respect
        # to the loss.

        # print([a.shape for a in activations])
        # print([ld.shape for ld in local_derivatives])
        # print([l['weights'].shape for l in self.layers])
        # print(loss.shape)

        activations = activations[:-1]
        
        delta = [np.multiply(local_derivatives[-1], loss)]
        derivatives = [np.matmul(activations[-1].T, delta[0])]

        local_derivatives = list(reversed(local_derivatives[:-1]))
        activations = list(reversed(activations[:-1]))
        layers = list(reversed(self.layers[1:]))
        
        for ii, layer in enumerate(layers):
            delta_l = (layer['weights'] @ delta[-1].T).T
            delta_l = np.multiply(local_derivatives[ii], delta_l)
            delta.append(delta_l)
            deriv = np.matmul(activations[ii].T, delta_l)
            if len(deriv.shape) == 1:
                deriv = deriv.reshape(1, -1)
            derivatives.append(deriv)

        del delta
        derivatives.reverse()
        return derivatives

    def compile(self, loss, optimizer):
        self.loss = get_loss(loss)
        if type(optimizer) == str:
            self.optimizer = get_optimizer(optimizer)
        self.compiled = True

    def fit(self, X, y, epochs: int=1):
        if not self.compiled:
            raise Exception('Model must be compiled prior to training')
        
        for epoch in range(1, epochs + 1):
            out, activations, local_derivatives = self._feed_forward(X, training=True)
            loss = self.loss(out, y, differentiate=True)
            derivatives = self._back_prop(
                loss, activations, local_derivatives
            )
            
            for ii, layer in enumerate(self.layers):
                layer['weights'] = optimizer.update(
                    layer['weights'], derivatives[ii]
                )

        return

    def predict(self, X):
        return self._feed_forward(X, training=False)


def run_tests():
    passed_tests = 0
    failed_tests = 0

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
            ]]),
            np.array([
                [outp * np.tanh(w11 * i)][0][0],
                [outp * np.tanh(w12 * i)][0][0]
            ])
        ]

    D = np.array(D)
    d = np.array(d)

    error = np.sum(np.sum(D - d))

    if error == 0.0:
        p = p + 1
    else:
        f = f + 1

    return p, f
    
if __name__ == '__main__':
    test1(1, 1, 42)
    np.random.seed(42)
    nn = NeuralNetwork(1)
    nn.add_layer(10, 'tanh', bias=False)
    nn.add_layer(20, 'tanh', bias=False)
    nn.add_layer(20, 'tanh', bias=False)
    nn.add_layer(2, 'sigmoid', bias=False)

    optimizer = SGD(learning_rate=0.03)
    nn.compile('mse', optimizer)

    # # nn = NeuralNetwork(1)
    # # nn.add_layer(2, 'tanh', bias=False)
    # # nn.add_layer(1, 'sigmoid', bias=False)
    # # print([l['weights'] for l in nn.layers])

    print(nn.predict(np.array([1])))
    nn.fit(np.array([1]), np.array([0.5, 0.3]), 10)
    print(nn.predict(np.array([1])))
    nn.fit(np.array([1]), np.array([0.5, 0.3]), 100)
    print(nn.predict(np.array([1])))
