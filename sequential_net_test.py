import numpy as np
import time

from sequential_net import Sequential, Dense, InputLayer
from optimizers import get_optimizer

def sequential_net_test__feed_forward(passed, failed):
    # Set seed for consistancy
    np.random.seed(42)
    
    nn = Sequential([
        InputLayer(3),
        Dense(4, activation='tanh'),
        Dense(1, activation='sigmoid'),
    ])

    nn.compile('mse', 'sgd')

    W1 = nn.layers[0].weights
    W2 = nn.layers[1].weights
    W3 = nn.layers[2].weights

    dt_pt = np.array([0.2, 0.1, 0.3])

    activs_cmp = []
    lds_cmp = []

    out_cmp = dt_pt @ W1
    activs_cmp.append(out_cmp)
    lds_cmp.append(np.ones(shape=(1, 3)))

    mat = out_cmp @ W2 + nn.layers[1].bias
    out_cmp = nn.layers[1].activation(mat)
    activs_cmp.append(out_cmp)
    lds_cmp.append(nn.layers[1].activation(mat, differentiate=True))

    mat = out_cmp @ W3 + nn.layers[2].bias
    out_cmp = nn.layers[2].activation(mat)
    activs_cmp.append(out_cmp)
    lds_cmp.append(nn.layers[2].activation(mat, differentiate=True))

    out, activs, lds = nn._feed_forward([dt_pt], training=True)

    activs_match = True
    for a, ac in zip(activs, activs_cmp):
        if np.any(a - ac != 0):
            activs_match = False

    lds_match = True
    for ld, ldc in zip(lds, lds_cmp):
        if np.any(ld-ldc != 0):
            lds_match = False

    if (out - out_cmp)[0][0] != 0 or not lds_match \
       or not activs_match:
        print("Failed single element _feed_forward test")
        failed += 1
    else:
        passed += 1

    dt_batch = np.array([[0.2, 0.1, 0.3],
                         [0.1, 0.45, 0.6],
                         [0.39, 0.04, 0.1]])

    out, activs, lds = nn._feed_forward(dt_pt, training=True)

    activs_cmp = []
    lds_cmp = []

    out_cmp = dt_batch @ W1
    activs_cmp.append(out_cmp)
    lds_cmp.append(np.ones(shape=(3, 3)))

    mat = out_cmp @ W2 + nn.layers[1].bias
    out_cmp = nn.layers[1].activation(mat)
    activs_cmp.append(out_cmp)
    lds_cmp.append(nn.layers[1].activation(mat, differentiate=True))

    mat = out_cmp @ W3 + nn.layers[2].bias
    out_cmp = nn.layers[2].activation(mat)
    activs_cmp.append(out_cmp)
    lds_cmp.append(nn.layers[2].activation(mat, differentiate=True))

    out, activs, lds = nn._feed_forward(dt_batch, training=True)

    activs_match = True
    for a, ac in zip(activs, activs_cmp):
        if np.any(a - ac != 0):
            activs_match = False

    lds_match = True
    for ld, ldc in zip(lds, lds_cmp):
        if np.any(ld-ldc != 0):
            lds_match = False

    if np.any(out - out_cmp != 0) or not lds_match \
       or not activs_match:
        print("Failed batched _feed_forward test")
        failed += 1
    else:
        passed += 1
    
    return passed, failed

def sequential_net_test__back_prop(passed, failed, seed=time.time()):
    np.random.seed(int(seed))
    
    nn = Sequential([
        InputLayer(3),
        Dense(4, activation='tanh'),
        Dense(1, activation='sigmoid'),
    ])

    nn.compile('mse', 'sgd')

    W1 = nn.layers[0].weights
    W2 = nn.layers[1].weights
    W3 = nn.layers[2].weights

    b1 = None
    b2 = nn.layers[1].bias
    b3 = nn.layers[2].bias

    def f(X):
        x, y, z = X
        return np.array([x + y + 0.5*z])

    dt_pt = np.array([0.2, 0.1, 0.3])

    out, activs, lds = nn._feed_forward([dt_pt], training=True)

    loss = nn.loss(out, f(dt_pt), differentiate=True)
    
    # deltas[0] : (1, 1)
    # deltas[1] : (1, 4)
    # deltas[2] : (1, 3)

    # derivs[0] : (4, 1)
    # derivs[1] : (3, 4)
    # derivs[2] : (3, 3)

    deltas_cmp = [np.multiply(loss, lds[2])]
    derivs_cmp = [np.matmul(deltas_cmp[0].T, activs[1]).T]

    deltas_cmp.append(np.matmul(
        deltas_cmp[0], np.multiply(lds[1], W3.T)
    ))
    derivs_cmp.append(np.matmul(deltas_cmp[1].T, activs[0]).T)
    
    deltas_cmp.append(np.matmul(
        deltas_cmp[1], np.multiply(lds[0], W2.T)
    ))
    derivs_cmp.append(np.matmul(deltas_cmp[2].T, [dt_pt]).T)

    deltas = nn._back_prop(loss, activs, lds)

    deltas_match = True
    for d, dc in zip(deltas, deltas_cmp):
        if np.any(np.abs(d - dc) > 1e-16):
            # Evidently I can get a small error since I am computing
            # deltas_cmp differently. 1e-16 error seems tolerable...
            deltas_match = False

    if deltas_match:
        passed += 1
    else:
        print("Failed unbatched _back_prop test")
        failed += 1
        
    return passed, failed

def sequential_net_test_training():
    from optimizers import SGD
    
    nn = Sequential([
        InputLayer(num_features),
        Dense(3, activation='tanh'),
        Dense(2, activation='sigmoid'),
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
    
    nn.fit(
        data,
        y,
        epochs=40,
        batch_size=1
    )

    X_test = [np.array([1, 0.2, 0.1])]
    print(nn.predict(X_test))


def dense_test_update(passed, failed):
    prev = Dense(2)
    dense = Dense(4, activation='tanh')

    dense.compile(prev)

    activ = np.array([[0.2, 0.1]])
    delta = np.array([[0.1, 0.3, -0.1, -0.2]])
    deriv = activ.T @ delta

    opt = get_optimizer('sgd')

    print(dense.weights)

    dense.update(delta, activ, opt)

    print(dense.weights)

    return passed, failed
    
    

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        sequential_net_test__feed_forward,
        sequential_net_test__back_prop,
        dense_test_update,
    ]

    for test in tests:
        passed, failed = test(passed, failed)

    print("Passed: %d" % passed)
    print("Failed: %d" % failed)
    print("Total : %d" % (passed + failed))
