import theano
from theano import gof
from theano.tensor.basic import as_tensor_variable
import numpy as np
import theano.tensor as T
import theano.gradient

class MulGrad(gof.Op):
    __props__ = ("name",)

    def __init__(self, name):
        self.name = name

    def make_node(self, inp, factor):
        inp = theano.tensor.as_tensor_variable(inp)
        factor = theano.tensor.as_tensor_variable(factor)
        return theano.Apply(self, [inp, factor], [inp.type()])

    def perform(self, node, inputs, output_storage, params=None):
        output_storage[0][0] = inputs[0] + 0*inputs[1]

    def grad(self, inputs, gout):
        (ginp, ) = gout
        (inp_v, factor_v) = inputs
        factor_v = theano.tensor.as_tensor_variable(factor_v)
        return [factor_v*ginp, T.zeros_like(factor_v)]

    def infer_shape(self, nodes, shapes):
        return [shapes[0]]

mulgrad = MulGrad(name='mulgrad')

if __name__=="__main__":
    """
    print "ello"
    w, x, y = T.scalar('w'), T.scalar('x'), T.scalar('y')
    z1 = mulgrad(w*x, y)
    z2 =         w*x
    f = theano.function([w, x, y], [z1, z2])
    print f(4,5,0.5)
    gr1 = T.grad(mulgrad(w*x, y),x)
    gr2 = T.grad(        w*x,    x)
    g = theano.function([w, x, y], [gr1, gr2])
    print g(4,5,0.5)
    t = T.grad(mulgrad(w*x, y),y)
    h = theano.function([w, x, y], t, on_unused_input='ignore')
    print f,g,h
    print h(4,5,0.5)
    """
    w, x, y = T.scalar('w'), T.scalar('x'), T.scalar('y')
    z1 = T.switch(T.isnan(w/x) + T.isinf(w/x), 0, w/x)
    z2 = w/x
    f = theano.function([w, x, y], [z1, z2], on_unused_input='ignore')
    print f(0,0,0.5)
    gr1 = T.grad(z1, x)
    gr2 = T.grad(z2, x)
    g = theano.function([w, x, y], [gr1, gr2], on_unused_input='ignore')
    print g(1,0,0.5)