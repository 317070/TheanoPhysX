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

grads = theano.grad(loss, parameters)
#grads = lasagne.updates.total_norm_constraint(grads, 1.0)

#grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
#theano_to_print.append(grad_norm)
learning_rate = theano.shared(np.float32(1.0))
updates.update(lasagne.updates.adam(grads, parameters, learning_rate=learning_rate, beta1=0.5, beta2=0.5))  # we maximize fitness
print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             []
                             + [fitness]
                             #+ [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters]
                             + theano_to_print
                             #+ T.grad(loss, all_parameters)
                             #+ all_parameters
                             ,
                             updates=updates,
                             )

iter_test = theano.function([],[fitness])
with open("theano-function.pkl", 'wb') as f:
    pickle.dump(iter_train, f, pickle.HIGHEST_PROTOCOL)

#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])

PARAMETERS_FILE = "optimized-parameters.pkl"

print "Running since %s..." % strftime("%H:%M:%S", localtime())

total_time = 5
last_result = None
avg_growth = 1.0
best = None
print iter_test()
iters = 1
results = iter_train()
learning_schedule = {
    1:2.0,
    5:1.0,
    50:0.1,
    75:0.05,
    100:0.01
}
import time
t = time.time()
while results[0][0]>0.02:
    if iters in learning_schedule:
        learning_rate.set_value(learning_schedule[iters])
    results = iter_train()
    iters += 1
    print iters, results, parameters[0].get_value()

print time.time() - t
# 1135

#tries: 1492: learning_rate=0.01, beta1=0.9, beta2=0.99
#tries: 672: learning_rate=0.3, beta1=0.3, beta2=0.99
#tries: learning_rate=1.0, beta1=0.5, beta2=0.99

#best:88

print iters
print "Finished on %s..." % strftime("%H:%M:%S", localtime())