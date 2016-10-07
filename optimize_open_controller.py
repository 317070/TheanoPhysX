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
engine.load_robot_model("robotmodel/full_predator.json")
spine_id = engine.getObjectIndex("spine")
BATCH_SIZE = 1
engine.compile(batch_size=BATCH_SIZE)
#engine.randomizeInitialState(rotate_around="spine")


# step 2: build the model, controller and engine for simulation
total_time = 0.01


def sign_rule(loss_or_grads, params, learning_rate):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * T.sgn(grad)

    return updates


def build_objectives(states_list):
    time, positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return (positions[-1,:,spine_id,:2] - engine.getInitialState()[0][:,spine_id,:2]).norm(L=2,axis=1)


def build_controller():
    p = np.float32(np.pi * 2 * 1.5)
    parameters = [
                  theano.shared( 0.0*np.ones(shape=(16,), dtype='float32')),
                  theano.shared( 0.0*np.ones(shape=(16,), dtype='float32')),
                  ]

    # given time and period, find periodic spline with control points in parameters
    def spline(time):
        phase = time * p + parameters[1]
        amplitude = parameters[0]
        signals = amplitude * T.sin(phase)
        signals = signals # * T.ones(shape=(16,))
        return signals[None,:]

    return {
        "output": spline,
        "shared_variables": parameters,
        "parameters": parameters
    }


def build_model():
    parameters = [T.scalar(name='totaltime', dtype='int32')]
    def get_shared_variables():
        return controller_parameters + engine.getSharedVariables()

    def control_loop(state, time):
        positions, velocities, rot_matrices = state
        time = time + engine.DT
        #sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        #controller["input"].input_var = sensor_values
        motor_signals = controller["output"](time)
        return (time,) + engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)

    outputs, updates = theano.scan(
        fn=lambda t,a,b,c,*ns: control_loop(state=(a,b,c), time=t),
        outputs_info=(np.float32(0),)+engine.getInitialState(),
        n_steps=parameters[0], #int(total_time/engine.DT),
        strict=True,
        non_sequences=get_shared_variables(),
    )
    assert len(updates)==0
    return parameters, outputs, controller_parameters, updates



controller = build_controller()

if args.compile:
    controller_parameters = controller["parameters"]

    parameters, states, all_parameters, updates = build_model()
    fitness = build_objectives(states)

    #import theano.printing
    #theano.printing.debugprint(T.mean(fitness), print_type=True)
    print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
    loss = -T.mean(T.switch(T.eq(fitness,np.NaN), 0, fitness))
    #loss = -T.mean(fitness)

    grads = theano.grad(loss, all_parameters)
    #grads = lasagne.updates.total_norm_constraint(grads, 1.0)

    #grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
    #theano_to_print.append(grad_norm)
    updates.update(sign_rule(grads, all_parameters, 0.1))  # we maximize fitness
    print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
    iter_train = theano.function(parameters,
                                 []
                                 + [fitness]
                                 #+ [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters]
                                 + theano_to_print
                                 #+ T.grad(loss, all_parameters)
                                 #+ all_parameters
                                 ,
                                 updates=updates,
                                 )

    iter_test = theano.function(parameters,[fitness])
    with open("theano-function.pkl", 'wb') as f:
        pickle.dump(iter_train, f, pickle.HIGHEST_PROTOCOL)
else:
    print "Loading theano function since %s..." % strftime("%H:%M:%S", localtime())
    with open("theano-function.pkl", 'rb') as f:
        iter_train = pickle.load(f)
        iter_test = None

#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])

PARAMETERS_FILE = "optimized-parameters.pkl"
def load_parameters():
    with open(PARAMETERS_FILE, 'rb') as f:
        resume_metadata = pickle.load(f)
        for p,val in zip(controller["parameters"],resume_metadata['param_values']):
            p.set_value(val)
    return "Finished"

def dump_parameters():
    with open(PARAMETERS_FILE, 'wb') as f:
        pickle.dump({
            'param_values': controller["parameters"]
        }, f, pickle.HIGHEST_PROTOCOL)

def are_there_NaNs(result):
    return np.isfinite(result).all() \
        and all([np.isfinite(p).all() for p in controller["parameters"]])

if not args.restart:
    print "Loading parameters... ", load_parameters()
else:
    dump_parameters()

print "Running since %s..." % strftime("%H:%M:%S", localtime())

total_time = 10
last_result = None
avg_growth = 1.0
target = np.mean(iter_test(500)[0])/5.0
best = None

while total_time<=1000:
    results = iter_train(total_time)
    speed = np.mean(results[0]) / total_time

    testspeed = np.mean(iter_test(500)[0])/5.0
    if testspeed>best:
        best = testspeed
        dump_parameters()

    if last_result != None:
        growth = np.mean(results[0]) - last_result
        avg_growth = 0.1*growth+0.9*avg_growth

    print "%.6f\t%.4f\t%d\t%.3f/%.3f\t%.3f" % (speed, avg_growth, total_time, testspeed,target,best)

    if avg_growth<0.001:
        total_time += 1

    last_result = np.mean(results[0])


print "Finished on %s..." % strftime("%H:%M:%S", localtime())