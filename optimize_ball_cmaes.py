from collections import OrderedDict
import math
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
from BatchTheanoPhysicsSystem import BatchedTheanoRigid3DBodyEngine
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine, theano_to_print
import lasagne
import sys
import numpy as np
from time import strftime, localtime
import datetime
import cPickle as pickle
import argparse
import cma

# initial: [ 0.42198306,  0.3469744 ,  0.57974786,  0.38210401]

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--compile', dest='compile',
                     help='re-compile the theano function',
                     action='store_const', const=True, default=False)
parser.add_argument('--restart', dest='restart',
                     help='have new random parameters',
                     action='store_const', const=True, default=False)
args = parser.parse_args()
sys.setrecursionlimit(10**6)

print "Started on %s..." % strftime("%H:%M:%S", localtime())
import random
random.seed(0)
np.random.seed(0)

# step 1: load the physics model
engine = BatchedTheanoRigid3DBodyEngine()
engine.load_robot_model("robotmodel/ball.json")
spine_id = engine.getObjectIndex("ball")
BATCH_SIZE = 1
engine.compile(batch_size=BATCH_SIZE)
#engine.randomizeInitialState(rotate_around="spine")


# step 2: build the model, controller and engine for simulation
total_time = 5


def sign_rule(loss_or_grads, params, learning_rate):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * T.sgn(grad)

    return updates


def build_objectives(states_list):
    time, positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return (positions[-1,:,spine_id,:] - np.array([10.0,0.0,0.5],dtype='float32')).norm(L=2,axis=1) + (velocities[-1,:,spine_id,:]).norm(L=2,axis=1)


def build_model():
    parameters = []

    def get_shared_variables():
        return engine.getSharedVariables()

    def control_loop(state, time):
        positions, velocities, rot_matrices = state
        time = time + engine.DT

        return (time,) + engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=[])

    outputs, updates = theano.scan(
        fn=lambda t,a,b,c,*ns: control_loop(state=(a,b,c), time=t),
        outputs_info=(np.float32(0),)+engine.getInitialState(),
        n_steps=int(total_time/engine.DT),
        strict=True,
        non_sequences=get_shared_variables(),
    )
    assert len(updates)==0
    parameters.append(engine.getInitialState()[1])
    return parameters, outputs, updates



parameters, states, updates = build_model()
fitness = build_objectives(states)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
loss = T.mean(T.switch(T.eq(fitness,np.NaN), 0, fitness))
#loss = -T.mean(fitness)


#grads = lasagne.updates.total_norm_constraint(grads, 1.0)

#grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
#theano_to_print.append(grad_norm)
print "Compiling since %s..." % strftime("%H:%M:%S", localtime())

iter_test = theano.function([],[fitness])

print "Running since %s..." % strftime("%H:%M:%S", localtime())

options = {'ftarget':2e-2, 'seed':1}
iters = 0

def iter_test_safe(param):
    global iters
    parameters[0].set_value(np.array([[param]],dtype='float32'))
    res = iter_test()[0]
    iters += 1
    print iters, res, param
    return res

import time
t = time.time()
print cma.fmin(iter_test_safe, parameters[0].get_value()[0][0], sigma0=1.0, options=options)
print time.time() - t

print "Finished on %s..." % strftime("%H:%M:%S", localtime())