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

    batches = []
    for ii in range(slices):
        batches.append(
            Z[ii * batch_size:(ii + 1) * batch_size]
        )
    if len(X) % batch_size != 0:
        batches.append(X[slices * batch_size:])
    return batches

def sum_matricies(X: list):
    s = X[0]
    for x in X[1:]:
        s += x
    return s

if __name__ == '__main__':
    print(make_batches(list('a'*30), 4))
