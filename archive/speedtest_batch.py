import theano
import theano.tensor as T
import numpy as np
import time
import random
from operator import mul




def test_matrix_matrix_el():
    x = T.tensor4("x",dtype='float32')
    y = T.tensor4("y",dtype='float32')

    vars = [x,y]
    f0 = theano.function(vars, T.sum(x * y, axis=(2,3)))
    f1 = theano.function(vars, T.batched_dot(x.reshape(shape=(40*50, 60*70)),
                                             y.reshape(shape=(40*50, 60*70))
                                             ).reshape(shape=(40,50)))

    fs = [f0,f1]
    test(vars,fs)


def test_vector_matrix():
    x = T.matrix("x",dtype='float32')
    y = T.tensor3("y",dtype='float32')

    vars = [x,y]

    f0 = theano.function(vars, T.sum(x[:,:,None] * y[:,:,:], axis=-2))
    f1 = theano.function(vars, T.batched_dot(x[:,None,:], y)[:,0,:])
    f2 = theano.function(vars, T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)]))

    fs = [f0,f1,f2]
    test(vars,fs)


def test_matrix_matrix():
    x = T.tensor3("x",dtype='float32')
    y = T.tensor3("y",dtype='float32')

    vars = [x,y]

    f0 = theano.function(vars, T.sum(x[:,:,:,None] * y[:,None,:,:], axis=-2))
    f1 = theano.function(vars, T.batched_dot(x, y))
    f2 = theano.function(vars, T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)]))

    fs = [f0,f1,f2]
    test(vars,fs)

def test(vars,fs):
    for _ in xrange(100):
        shapes = []
        for var in vars:
            while True:
                #s = [random.randint(1,500) for i in xrange(var.ndim)]
                s = [40,50,60,70]
                if np.prod(s)<1e7: #limit tensor size
                    break
            shapes.append(s)

        shapes[1] = shapes[0]

        print shapes
        arguments = []
        for shape in shapes:
            arguments.append(np.float32(np.arange(reduce(mul, shape, 1)).reshape(shape)))

        nr = None
        for idx,f in enumerate(fs):
            start = time.time()
            for __ in xrange(100):
                r = f(*arguments)
            print "f%d-time: %.3fs" % (idx, time.time() - start)

            if nr is None:
                nr = r
            else:
                assert np.isclose(r, nr, rtol=1.e-3, atol=1.e-3).all()
        print


test_matrix_matrix_el()
#test_matrix_matrix()