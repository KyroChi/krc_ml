import numpy as np
from activations import get_activation
from losses import get_loss
from optimizers import Optimizer, get_optimizer
from initializers import get_initializer
from preprocessing import make_batches, sum_matricies

class Layer(object):
    """
    Primative Layer class.
    """
    def __init__(self, name: str=None):
        pass

    def compile(self, previous_layer):
        raise NotImplementedError(
            "Cannot call 'compile' on primative 'Layer' object"
        )

    def evaluate(self):
        raise NotImplementedError(
            "Cannot call 'evaluate' on primative 'Layer' object"
        )

    def update(self, delta_l, deriv, optimizer):
        raise NotImplementedError(
            "Cannot call 'update' on primative 'Layer' object"
        )

    def get_weights(self):
        raise NotImplementedError(
            "Cannot call 'get_weights' on primative 'Layer' object"
        )

    def set_weights(self):
        raise NotImplementedError(
            "Cannot call 'get_weights' on primative 'Layer' object"
        )

class InputLayer(Layer):
    """
    Handles input to the sequential neural net.

    Copies it's inputs to it's outputs.
    """
    def __init__(
            self,
            input_shape
    ):
        self.input_size = input_shape
        self.output_size = input_shape

        self.weights = np.eye(input_shape)

    def compile(self, *args, **kwargs):
        return

    def evaluate(self, X, training):
        if training:
            return X, np.ones(shape=(len(X), 1))
        else:
            return X

    def update(self, *args, **kwargs):
        return

class Dropout(Layer):
    """
    Drops some number of the connections between layers during 
    training.
    
    Does nothing on non-training evaluation. Adds minimal 
    computational overhead when not training.
    """
    def __init__(self, alpha):
        if alpha < 0 or alpha >= 1:
            raise Exception("Dropout alpha must be in [0, 1)")
        self.alpha = alpha
        
        self.output_size = None
        self.input_size = None
        self.weights = None

    def compile(self, previous_layer):
        self.output_size = previous_layer.output_size
        self.input_size = previous_layer.output_size
        self.weights = np.eye(self.output_size)
        return

    def evaluate(self, X, training):
        if training:
            drop = np.eye(X.shape[-1])
            for ii in range(X.shape[-1]):
                r = np.random.uniform()
                if r < self.alpha:
                    drop[ii, ii] = 0
                    
            # May have to change the derivative
            return np.matmul(X, drop), np.ones(shape=X.shape[-1])
        else:
            return X

    def update(self, *args, **kwargs):
        return

class Dense(Layer):
    """
    Fully connected layer.
    """
    def __init__(
            self,
            width,
            activation='tanh',
            initializer='glorot',
            bias=True,
    ):
        self.activation = get_activation(activation)
        self.initializer = get_initializer(initializer)

        self.output_size = width;
        self.input_size = None;
        
        if bias:
            self.bias = self.initializer(1, self.output_size)
        else:
            self.bias = np.zeros(shape=(1, self.output_size))

        self.compiled = False
        self.weights = None

    def compile(self, previous_layer):
        # Needs things like previous layer size
        input_size = previous_layer.output_size
        self.input_size = input_size
        self.weights = self.initializer(
            self.input_size, self.output_size
        )
        return

    def evaluate(self, X, training):
        batch_size = len(X)

        if batch_size > 1:
            bias = np.stack([self.bias] * batch_size, axis=1)[0]
        else:
            bias = self.bias

        mat = X @ self.weights + bias
        out = self.activation(mat)

        if training:
            return out, self.activation(
                mat, differentiate=True
            )
        else:
            return out

    def update(self, delta, prev_activ, optimizer):
        self.bias = optimizer.update(self.bias, delta)
        self.weights = optimizer.update(
            self.weights, np.matmul(delta, prev_activ.T)
        )
        
        return

class Sequential(object):
    def __init__(self, layers: list):
        for layer in layers:
            if not issubclass(type(layer), Layer):
                raise Exception(
                    "Layer list must contain type 'Layer'."
                )
            
        self.layers = layers if layers else []
        self.compiled = False
        
        self.optimizer = None

    def add_layer(
            self,
            layer
    ):
        if type(layer) is not Layer:
            raise Exception("Added layer must be type 'Layer'.")
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = get_loss(loss)
        if type(optimizer) == str:
            self.optimizer = get_optimizer(optimizer)
        elif issubclass(type(optimizer), Optimizer):
            self.optimizer = optimizer
        else:
            raise Exception(
                "Optimizer must be type 'str' or 'Optimizer'"
            )

        for ii, layer in enumerate(self.layers[1:]):
            # First layer must be an input layer, which will not
            # be compiled.
            layer.compile(self.layers[ii])
            
        self.compiled = True

    def fit(
            self,
            X, y,
            epochs: int=1,
            batch_size: int=None,
            verbose=True
    ):
        if not self.compiled:
            raise Exception('Model must be compiled prior to training')

        if not len(X) == len(y):
            raise Exception(
                'X length %d does not match y length %d.' % (
                    len(X), len(y)
                )
            )

        # Handle mini-batching
        if batch_size is None:
            batch_size = 1
            
        batches_X, batches_y = make_batches(X, y, batch_size)
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            
            for b_X, b_y in zip(batches_X, batches_y):
                
                # Desired Sequential API
                out, activs, lds = self._feed_forward(
                    b_X, training=True
                )

                epoch_loss += self.loss(out, b_y)
                
                loss = self.loss(out, b_y, differentiate=True)
                
                deltas = self._back_prop(loss, activs, lds)

                for dd in deltas:
                    dd = dd.mean(axis=0)

                for activ in activs:
                    activ = activ.mean(axis=0)

                # activs.insert(0, b_X)

                for layer, delta, activ in zip(
                        self.layers,
                        reversed(deltas),
                        activs
                ):
                    # Optimizer updates weights
                    layer.update(
                        delta.mean(axis=0),
                        activ.mean(axis=0),
                        self.optimizer
                    )

            if verbose:
                # print(self.layers[1].weights)
                print(
                    "Finished epoch %d with loss %.5f" % (
                        epoch, epoch_loss / batch_size)
                )
                    
        return

    def predict(self, X):
        return self._feed_forward(X, training=False)

    def _feed_forward(self, X, training=False):
        if training:
            activations = []
            derivatives = []
            
        out = X
        for layer in self.layers:
            if training:
                out, ld = layer.evaluate(out, training=True)
                activations.append(out)
                derivatives.append(ld)
            else:
                out = layer.evaluate(out, training=False)
                
        if training:
            return out, activations, derivatives
        else:
            return out

    def _back_prop(self, loss, activations, local_derivatives):
        delta = [np.multiply(local_derivatives[-1], loss)]
        derivatives = [np.matmul(delta[0], activations[-1].T)]
        
        local_derivatives = list(reversed(local_derivatives[:-1]))
        activations = list(reversed(activations[:-1]))

        for ii, layer in enumerate(list(reversed(self.layers[1:]))):
            delta_l = delta[-1] @ layer.weights.T
            delta_l = np.multiply(delta_l, local_derivatives[ii])
            
            delta.append(delta_l)

        return delta
    

if __name__ == "__main__":
    from optimizers import SGD
    np.random.seed(42)

    num_features = 3

    nn = Sequential([
        InputLayer(num_features),
        Dense(5, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(2, activation='tanh'),
    ])

    optimizer = SGD(learning_rate=0.1)
    nn.compile('mse', optimizer)

    def f(X):
        x, y, z = X
        return np.array([0.5*x + 0.2*y, 0.3*z])

    dpts = 500
    data = [
        [np.random.rand() for _ in range(num_features)
        ] for _ in range(dpts)]
    y = [f(d) for d in data]

    X_test = [np.array([1, 0.2, 0.1])]
    print(nn.predict(X_test))
    
    nn.fit(
        data,
        y,
        epochs=40,
        batch_size=1
    )

    print(nn.predict(X_test))
    print(f(X_test[0]))
