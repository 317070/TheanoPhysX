import numpy as np
import theano
import theano.tensor as T


def theano_dot_last_dimension_matrices(x, y):
    if x.ndim==3 and y.ndim==3:
        if ("theano" in str(type(x)) and x.broadcastable[0] == False)\
                or ("numpy" in str(type(x)) and x.shape[0] != 1):
            return T.batched_dot(x, y)
        else:
            return T.tensordot(x[0,:,:], y, axes=[[1], [1]])
    else:
        return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)])

def theano_dot_last_dimension_vectors(x, y):
    return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-1,)])

def theano_dot_last_dimension_vector_matrix(x, y):
    if x.ndim==2 and y.ndim==3:
        if ("theano" in str(type(x)) and x.broadcastable[0] == False)\
                or ("numpy" in str(type(x)) and x.shape[0] != 1):
            return T.batched_dot(x[:,None,:], y)[:,0,:]
        else:
            return T.tensordot(x[0,:], y, axes=[[0], [1]])
    else:
        return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)])

def theano_dot_last_dimension_matrix_vector(x, y):
    return T.batched_tensordot(x, y, axes=[(x.ndim-1,), (y.ndim-1,)])


def theano_stack_batched_integers_mixed_numpy(L, expected_shape=None):
    """
    It is more efficient to stack in numpy than in theano. So stack numpy's first, than stack them all at once with Theano.
    """
    r = []
    last_numpy = None
    for l in L:
        if "theano" in str(type(l)):
            if last_numpy is not None:
                r.append(last_numpy.astype('float32'))
                last_numpy = None
            if expected_shape is None:
                l = l.dimshuffle(0,'x')
            r.append(l)
        else:
            if expected_shape is None:
                l = l[:,None]
            if last_numpy is None:
                last_numpy = l
            else:
                last_numpy = np.concatenate([last_numpy, l], axis=(1 if expected_shape is None else 0))
    if last_numpy is not None:
        r.append(last_numpy.astype('float32'))

    if expected_shape is None:
        result = T.concatenate(r, axis=1)
    else:
        result = T.concatenate(r, axis=0).reshape(expected_shape)
    return result


def single_skew_symmetric(x):
    a,b,c = x[:,0,None,None],x[:,1,None,None],x[:,2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)

def batch_skew_symmetric(x):
    a,b,c = x[:,:,0,None,None],x[:,:,1,None,None],x[:,:,2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)


def theano_convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    return theano_dot_last_dimension_vector_matrix(coor, rot_matrix)
    #return T.sum(rot_matrix * coor[:,:,None], axis=1)

def theano_convert_model_to_world_coordinate(coor, rot_matrix, pos_vectors):
    return theano_dot_last_dimension_vector_matrix(coor, rot_matrix) + pos_vectors
    #return T.sum(rot_matrix * coor[:,:,None], axis=1)

def theano_convert_world_to_model_coordinate_no_bias(N, s_rot_matrices):
    raise NotImplementedError()