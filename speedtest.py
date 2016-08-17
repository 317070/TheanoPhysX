import theano
import theano.tensor as T
import numpy as np
import time
import random

x = T.tensor4("x",dtype='float32')

f1 = theano.function([x], theano.scan(lambda y: T.tensordot(y, y, axes=([1, 2], [1, 2])), sequences=x)[0])
f2 = theano.function([x], T.sum(x[:, :, None, :,:] * x[:, None, :, :,:], axis=(-2,-1)))

for _ in xrange(100):
    while True:
        s = [random.randint(1,100) for i in xrange(4)]
        if np.prod(s)<1e7:
            break

    x = np.float32(np.arange(np.prod(s)).reshape(s))
    print "shape:",s

    start = time.time()
    for i in xrange(100):
        r1 = f1(x)
    print "f1-time: %.3fs" % (time.time() - start)

    start = time.time()
    for i in xrange(100):
        r2 = f2(x)
    print "f2-time: %.3fs" % (time.time() - start)
    assert np.isclose(r1, r2).all()
    print

