from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
def relu(inp):
    return np.maximum(0,inp)

def nrelu(inp):
    return np.minimum(0,inp)
Z1 = X
Z2 = Y
a = Z1+Z2
b = abs(Z1-Z2)+1
#Z = (relu(a+b) - relu(a-b)) / (2*b)

Z = relu(X-relu(Y)*relu(X)) + nrelu(X+relu(Y)*relu(-X))

def trect(inp):
    return T.nnet.relu(inp)
#"""
tx = T.scalar()
ty = T.scalar()
ta = tx+ty
tb = abs(tx-ty) + 1
tz = (trect(ta+tb) - trect(ta-tb)) / (2*tb)

tg = T.grad(tz, wrt=[tx,ty])
tgn = (tg[0]**2 + tg[1]**2)**0.5

func = theano.function([tx,ty],tz,allow_input_downcast=True)
func_gn = theano.function([tx,ty],tgn,allow_input_downcast=True)
#"""
print X.ndim

Z = [[func(X[i,j],Y[i,j]) for i in xrange(len(X))] for j in xrange(len(Y))]
Z1 = [[func_gn(X[i,j],Y[i,j]) for i in xrange(len(X))] for j in xrange(len(Y))]

# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, alpha=0.5)
surf2 = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                      linewidth=0, alpha=0.5)
#cont = ax.contour(X, Y, Z, 15, linewidths = 0.5, colors = 'k')
#ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('ht axis')
ax.set_ylabel('B axis')
ax.set_zlabel('Z axis')
plt.show()