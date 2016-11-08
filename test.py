import numpy as np

BATCH_SIZE = 1000

def sample():
    res = np.array([1.]*3*BATCH_SIZE).reshape((BATCH_SIZE,3))
    while (np.sum(res**2, axis=-1) > 1).any():
        idx = (np.sum(res**2, axis=-1) > 1)
        s = np.concatenate([
            np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
            np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
            np.random.uniform(low=0.0, high=1.0, size=(BATCH_SIZE,1))
            ], axis=1).astype('float32')
        res[idx] = s[idx]
    res += np.array([0., 0., 0.1]*BATCH_SIZE, dtype='float32').reshape((BATCH_SIZE,3))
    return res.astype('float32')

import matplotlib.pyplot as plt

x = sample()
plt.scatter(x[:,0], x[:,1])
plt.show()
plt.scatter(x[:,1], x[:,2])
plt.show()
plt.scatter(x[:,0], x[:,2])
plt.show()
