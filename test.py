import numpy as np
import scipy.linalg

num_constraints = 3

M = np.random.rand(*(2,6,6))  # 0 constraints x 2 objects x 6 states x 6 states
J = np.random.rand(*(2,6))  # 0 constraints x 2 objects x 6 states


#M = np.ones((2,6,6))  # 0 constraints x 2 objects x 6 states x 6 states
#J = np.ones((2,6))  # 0 constraints x 2 objects x 6 states

mass = scipy.linalg.block_diag(M[0,:,:], M[1,:,:])
Jl = np.concatenate([J[0],J[1]])[None,:]
print mass
print Jl.shape

print np.dot(Jl,np.dot(mass,Jl.T))

B = np.sum(np.sum(J[:,None,:]*M, axis=-1)*J)
print B