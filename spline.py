'''useful function to set up circular cubic Hermite splines.
These are splines which repeat themselves every 2*pi, so you can call them just as you would call the sin() function
The splines will go through a predefined set of points at given times.

e.g.
>>>myspline = CircularSpline([0,1,2,15], [4,5,6,7])
will return a spline with the properties:
    myspline(t) = myspline(t+2*pi) for all t
    myspline(0) = 4
    myspline(1) = 5
    myspline(2) = 6
    myspline(15) = 7
You evaluate them also this way.
>>>print myspline(15)
        7
'''

import numpy as np
import math

class Sine:
    def __init__(self, phase, amplitude, offset):
        self.phase = phase
        self.amplitude = amplitude
        self.offset = offset

    def __call__(self,  t):
        return 500 - (self.amplitude * math.sin(t + self.phase) + self.offset)


'''Make a new basic circular spline object. Tangents are calulated using the Catmull-Rom spline method
this is a very smooth spline, with continuous tangents.'''
class CircularSpline(object):
    def __init__(self,  x,  y):
        np.seterr(all='warn')
        x = np.array(x,dtype=float)
        y = np.array(y,dtype=float)

        for i in range(len(x)):
            x[i] = x[i] % (2*np.pi)
        # sorteer op x-waarden
        temp = zip(x, y)
        temp.sort()
        x , y = zip(*temp)

        i=0
        while i < len(x):
            while abs(x[i-1]-x[i]) < 1e-8:
                x = np.delete(x, i)
                y = np.delete(y, i)
                if 1 == len(x):
                    break
                if i == len(x):
                    break
            i = i+1

        if abs(x[-1] - 2*np.pi - x[0]) < 1e-8:
            x = np.delete(x, len(x)-1)
            y = np.delete(y, len(y)-1)

        self.x = x
        self.y = y

        self._calculatem()
        self.once = False

    # Starting and ending tangents (using Catmull-Rom spline method)
    def _calculatem(self):
        self.m = np.zeros(len(self.x))
        for i in range(len(self.m)):
            p0 = np.mod(i-1, len(self.y))
            p1 = np.mod(i+1, len(self.y))
            x0 = self.x[p0]
            x1 = self.x[p1]
            if x0>=x1:
                x1 = x1+2*np.pi
            self.m[i] = (self.y[p1] - self.y[p0])   /   (x1-x0)

    def evaluate(self, xnew):
        #Find the neighbouring points to the x-location of the point we want to calculate
        xnew = np.mod(xnew,  2*np.pi)
        x=self.x
        y=self.y
        m=self.m
        found = False
        for ii in range(len(x)):
            if (x[ii] >= xnew):
                found = True
                break

        if(found):
            ii = ii-1
        # ii is now the index of the point before our point (this might be the last point)
        # Starting and ending data points
        if ii==-1:
            x0 = x[ii] - 2*np.pi
            x1 = x[ii+1]
            y0 = y[ii]
            y1 = y[ii+1]
            m0 = m[ii]
            m1 = m[ii+1]
        elif ii==len(x)-1:
            x0 = x[ii]
            x1 = x[0]+2*np.pi
            y0 = y[ii]
            y1 = y[0]
            m0 = m[ii]
            m1 = m[0]
        else:
            x0 = x[ii]
            x1 = x[ii+1]
            y0 = y[ii]
            y1 = y[ii+1]
            m0 = m[ii]
            m1 = m[ii+1]

        # Normalize to x_new to [0,1] interval
        h = (x1 - x0)
        t = (xnew - x0) / h
        if np.isnan([h, xnew, x0]).any() or h==0:
            if self.once:
                self.once=True
                print "m:", m
                print "h:", h
                print "x:", x
                print "y:", y
                print "xnew", xnew
                print "x0", x0

        # Compute the four Hermite basis functions
        h00 = ( 2.0 * t**3) - (3.0 * t**2) + 1.0
        h10 = ( 1.0 * t**3) - (2.0 * t**2) + t
        h01 = (-2.0 * t**3) + (3.0 * t**2)
        h11 = ( 1.0 * t**3) - (1.0 * t**2)

        ynew = (h00 * y0) + (h10 * h * m0) + (h01 * y1) + (h11 * h * m1)
        return ynew

    def __call__(self,  t):
        return self.evaluate(t)

    def plot(self, name='chart'):
        yvec = []
        xvec= np.arange(0, 10, 0.01)
        for xx in xvec:
            yvec.append(self(xx))
        import matplotlib.pyplot as plt
        #from matplotlib import rc
        #rc('text', usetex=True)
        #rc('ps', usedistiller='xpdf')
        #rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        #rc('font', family='serif')
        #rc('font', size='12.0')
        plt.figure()#figsize=(2, 2/1.618))
        prev_hold = plt.ishold()
        plt.hold(True)
        plt.plot(self.x, self.y, 'ro')
        plt.plot(xvec, yvec, 'b')
        plt.axvline(x=2*np.pi, color='r')
        plt.ylim([np.min(yvec)-0.1, np.max(yvec)+0.1])
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.hold(prev_hold)
        #plt.savefig(name +'.pdf', bbox_inches='tight')
        plt.draw()

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

'''This is a monotonic spline. It has the special property that local maxima and minima of your control points,
are also local maxima and minima of your spline, even after interpolation.
While it is less smooth, it still has continuous tangents.

It extends the class CircularSpline'''
class MonotonicSpline(CircularSpline):
    def _calculatem(self):
        x=self.x
        y=self.y
        self.m = np.zeros(len(x))
        secant = np.zeros(len(x))
        if x[0]+2*np.pi == x[-1]:
            secant[-1]=0
        else:
            secant[-1] = (y[0]-y[-1])/(x[0]+2*np.pi-x[-1])

        for i in range(len(self.m)):
            if i!=len(self.m)-1:
                if x[i+1]==x[i]:
                    secant[i] = 0
                else:
                    secant[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
            if (y[i]>=y[i-1] and y[i]>=y[(i+1)%len(self.m)]) or (y[i]<=y[i-1] and y[i]<=y[(i+1)%len(self.m)]):
                self.m[i] = 0
            else:
                self.m[i] = (secant[i-1]+secant[i])/2

        for i in range(len(self.m)):
            if secant[i] == 0:
                self.m[i] = 0
            else:
                alpha = self.m[i]/secant[i]
                beta = self.m[(i+1)%len(self.m)]/secant[i]
                if alpha**2+beta**2 > 9:
                    tau = 3/np.sqrt(alpha**2+beta**2)
                    if self.m[i]!=0:
                        self.m[i] = tau*alpha*secant[i]
                    if self.m[(i+1)%len(self.m)]!=0:
                        self.m[(i+1)%len(self.m)] = tau*beta*secant[i]

if __name__=="__main__":
    ##x = np.array([1.0, 2.5])
    ##y = np.array([-1.0, 1.0])
    #
    ##x = np.array([1.0, 1.5,  2.5, 5.2])
    ##y = np.array([-1.0, 0.4, 1.0, 0.2])
    #
    x = np.array([1.0, 2.0, 3.0, 4, 5.2, 5.9])
    y = np.array([-1.0, 0.4, -1.0, 1.0, 0.2, -0.8])

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    rc('ps', usedistiller='xpdf')
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('font', family='serif')
    rc('font', size='12.0')
    plt.figure(figsize=(6, 2/1.618))
    splines = [MonotonicSpline(x, y),CircularSpline(x, y)]
    for spline in splines:
        yvec = []
        xvec= np.arange(0, 13, 0.01)
        for xx in xvec:
            yvec.append(spline(xx))

        prev_hold = plt.ishold()
        plt.hold(True)
        plt.plot(x, y, 'ro')
        plt.plot(xvec, yvec, 'b')
        plt.axvline(x=2*np.pi, color='r')
        plt.ylim([np.min(yvec)-0.1, np.max(yvec)+0.1])
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.hold(prev_hold)
    #plt.savefig('spline.pdf', bbox_inches='tight')
    plt.show()