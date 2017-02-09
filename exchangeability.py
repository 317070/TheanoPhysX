import numpy as np
import scipy.stats

LENGTH = 256
DIMENSION = 3
rho = 0.5

# A = np.random.random_sample(size=(LENGTH,LENGTH))
# cov = np.dot(A.T,A)
cov = rho*np.ones(shape=(DIMENSION,DIMENSION)) + (1-rho)*np.eye(DIMENSION)
sample = np.random.multivariate_normal(mean=np.zeros(shape=(DIMENSION,)),
                              cov=cov,
                              size=1)[0]

print sample.shape


def R(x):
    return abs(x)

def Kalman(sample):
    # initialization
    x_hat = 0.  # initial estimate of the mean
    P = 1.  # initial precision of this x_hat
    #R = 1.  # observation noise

    likelihood = 0
    for observation in sample:

        likelihood += scipy.stats.multivariate_normal.logpdf(observation, mean=x_hat, cov=1/P)

        # we do observations with mean 0 and stdev 1
        y_tilde = observation-x_hat
        S = P + R(observation)
        K = P/S
        x_hat = x_hat + K*y_tilde
        P = (1-K)*P

        # predict
        mu = x_hat
        sigma = 1./P
        #print x_hat, P

    print "likelihood", likelihood
    return x_hat, P

for i in xrange(100):
    print
    print Kalman(sample)
    np.random.shuffle(sample)
    print scipy.stats.multivariate_normal.pdf(sample, mean=np.zeros(shape=(LENGTH,)),
                                    cov=cov,)



