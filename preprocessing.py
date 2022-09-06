import numpy as np

def make_batches(
        X: list,
        y: list,
        batch_size,
        shuffle=False
):
    """
    Split a list into batches.

    Assumes that X and y have already been checked for length match.
    """
    slices = len(X) // batch_size

    Z = list(zip(X, y))

    batch_X = []
    batch_y = []
    
    for ii in range(slices):
        batch_X.append(
            np.array(X[ii * batch_size:(ii + 1) * batch_size])
        )
        batch_y.append(
            np.array(y[ii * batch_size:(ii + 1) * batch_size])
        )
        
    if len(X) % batch_size != 0:
        batch_X.append(np.array(X[slices * batch_size:]))
        batch_y.append(np.array(y[slices * batch_size:]))
        
    return batch_X, batch_y

def sum_matricies(X: list):
    s = X[0]
    for x in X[1:]:
        s += x
    return s

if __name__ == '__main__':
    print(make_batches(list('a'*30), 4))
