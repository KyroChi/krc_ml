import numpy as np
from activations import get_activation
from losses import get_loss
from optimizers import get_optimizer
from preprocessing import make_batches, sum_matricies

class NeuralNetwork(object):
    """
    Implements a feedforward neural network which is trained using 
    backpropogation.,    
    """
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

    def fit(
            self,
            X, y,
            epochs: int=1,
            batch_size: int=None,
            verbose=False
    ):
        if not self.compiled:
            raise Exception('Model must be compiled prior to training')

        if not len(X) == len(y):
            # Assert that the inputs and targets have same first
            # dimension.
            raise Exception(
                'X length %d does not match y length %d.' % (
                    len(X), len(y)
                )
            )

        # Handle mini-batching
        if batch_size is None:
            batch_size = len(X)
        batch_X, batch_y = make_batches(X, y, batch_size)
        
        for epoch in range(1, epochs + 1):
            for (b_X, b_y) in zip(batch_X, batch_y):
                batch_derivatives = []
                for ii in range(len(b_X)):
                    x, y = b_X[ii], b_y[ii]
                    # TODO: Vectorize (i.e. run samples all at once)
                    # Batches can be parallized.
                    out, act, ld = self._feed_forward(
                        x, training=True
                    )
                    loss = self.loss(out, y, differentiate=True)

                    derivatives = self._back_prop(
                        loss, act, ld
                    )
                    batch_derivatives.append(derivatives)

                derivatives = sum_matricies(batch_derivatives)
            
                for ii, layer in enumerate(self.layers):
                    layer['weights'] = optimizer.update(
                        layer['weights'], derivatives[ii]
                    )

            if verbose:
                print('Completed epoch %d' % epoch)

        return

    def predict(self, X):
        return self._feed_forward(X, training=False)

    
if __name__ == '__main__':
    from optimizers import SGD
    np.random.seed(42)
    nn = NeuralNetwork(1)
    nn.add_layer(3, 'tanh', bias=False)
    nn.add_layer(10, 'tanh', bias=False)
    nn.add_layer(10, 'tanh', bias=False)
    nn.add_layer(2, 'sigmoid', bias=False)

    optimizer = SGD(learning_rate=0.05)
    nn.compile('mse', optimizer)

    # # nn = NeuralNetwork(1)
    # # nn.add_layer(2, 'tanh', bias=False)
    # # nn.add_layer(1, 'sigmoid', bias=False)
    # # print([l['weights'] for l in nn.layers])

    def f(x):
        x = x[0]
        return np.array([0.5*x, 0.3*x])

    dpts = 500
    data = [np.array([np.random.rand()]) for _ in range(dpts)]
    y = [f(d) for d in data]

    nn.fit(
        data,
        y,
        epochs=10,
        batch_size=1,
        verbose=True
    )
    print(nn.predict(np.array([1])))

    print(nn.layers[-2]['weights'])
