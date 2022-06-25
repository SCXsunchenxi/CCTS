import numpy as np
import logging
log = logging.getLogger(__name__)

def sgd(X, y_true, model, loss, batch_size=4, epochs=1, shuffle=False, lr=1e-3):
    for epoch in range(1, epochs+1):
        #log.info("EPOCH: {}".format(epoch))
        if shuffle:
            indices = np.random.permutation(X.shape[0])
        else:
            indices = np.arange(X.shape[0])
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[indices[start:end], :]
            y_batch = y_true[indices[start:end], :]
            sgd_step(X_batch.T, y_batch.T, model.next, loss, lr)
    return

def sgd_step(X, y, layer, loss, lr):
    if layer is None:
        log.debug("y_pred {}".format(X))
        log.debug("y_true {}".format(y))
        log.info("loss {}".format(loss.forward(X, y)))
        delta = loss.backward(X, y)
        log.debug("delta {}".format(delta))
        return delta
    else:
        log.debug(layer.__class__.__name__)
        log.debug("input {}".format(X))
        a = layer.forward(X)
        log.debug("output {}".format(a))
        delta = sgd_step(a, y, layer.next, loss, lr)
        log.debug("delta {}".format(delta))
        grad = layer.backward(X, delta)
        log.debug("grad \n{}".format(grad))
        if layer.trainable:
            layer.update(X, delta, lr)
            log.debug("NEW VALUES {}".format(layer.get_weights()))
        return grad

