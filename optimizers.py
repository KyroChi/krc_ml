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
