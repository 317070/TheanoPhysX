import theano
import theano.tensor as T
import numpy as np
import time
import random

x = T.tensor4("x",dtype='float32')

f0 = theano.function([x], theano.scan(lambda y: T.tensordot(y, y, axes=([1, 2], [1, 2])), sequences=x)[0])
f1 = theano.function([x], T.sum(x[:, :, None, :,:] * x[:, None, :, :,:], axis=(-2,-1)))
f2 = theano.function([x], T.batched_tensordot(x, x, axes=[(2,3), (2,3)]))
f3 = theano.function([x], T.batched_dot(x.flatten(3), x.flatten(3).dimshuffle(0,2,1)))

fs = [f0,f1,f2,f3]


for _ in xrange(100):
    while True:
        s = [random.randint(1,100) for i in xrange(4)]
        s[1] = 8
        if np.prod(s)<1e7:
            break

    x = np.float32(np.arange(np.prod(s)).reshape(s))
    print "shape:",s

    nr = None
    for idx,f in enumerate(fs):
        start = time.time()
        for i in xrange(100):
            r = f(x)
        print "f%d-time: %.3fs" % (idx, time.time() - start)
        if nr is None:
            nr = r
        else:
            assert np.isclose(r, nr, rtol=1.e-3, atol=1.e-3).all()
    print


