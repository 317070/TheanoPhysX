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
BATCH_SIZE = 128
engine.compile(batch_size=BATCH_SIZE)
engine.randomizeInitialState(rotate_around="spine")


# step 2: build the model, controller and engine for simulation
total_time = 10

def build_objectives(states_list):
    positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return (positions[-1,:,spine_id,:2] - engine.getInitialState()[0][:,spine_id,:2]).norm(L=2,axis=1)


def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,engine.num_sensors), name="sensor_values")
    #l_1 = lasagne.layers.DenseLayer(l_input, 1024,
    #                                     nonlinearity=lasagne.nonlinearities.rectify,
    #                                     W=lasagne.init.Orthogonal("relu"),
    #                                     b=lasagne.init.Constant(0.0),
    #                                     )
    #l_2 = lasagne.layers.DenseLayer(l_1, 1024,
    #                                     nonlinearity=lasagne.nonlinearities.rectify,
    #                                     W=lasagne.init.Orthogonal("relu"),
    #                                     b=lasagne.init.Constant(0.0),
    #                                     )
    l_result = lasagne.layers.DenseLayer(l_input, 16,
                                         nonlinearity=lasagne.nonlinearities.identity,
                                         W=lasagne.init.Orthogonal(),
                                         b=lasagne.init.Constant(0.0),
                                         )
    return {
        "input":l_input,
        "output":l_result
    }


def build_model():

    def get_shared_variables():
        return controller_parameters + engine.getSharedVariables()

    def control_loop(state):
        positions, velocities, rot_matrices = state
        sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        controller["input"].input_var = sensor_values
        motor_signals = lasagne.layers.helper.get_output(controller["output"])
        return engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)

    outputs, updates = theano.scan(
        fn=lambda a,b,c,*ns: control_loop(state=(a,b,c)),
        outputs_info=engine.getInitialState(),
        n_steps=int(math.ceil(total_time/engine.DT)),
        strict=True,
        non_sequences=get_shared_variables(),
    )
    assert len(updates)==0
    return outputs, controller_parameters, updates



controller = build_controller()

if args.compile:
    controller_parameters = lasagne.layers.helper.get_all_params(controller["output"])

    states, all_parameters, updates = build_model()
    fitness = build_objectives(states)

    #import theano.printing
    #theano.printing.debugprint(T.mean(fitness), print_type=True)
    print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
    loss = -T.mean(T.switch(T.eq(fitness,np.NaN), 0, fitness))

    grads = theano.grad(loss, all_parameters)
    grads = lasagne.updates.total_norm_constraint(grads, 1.0)

    #grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
    #theano_to_print.append(grad_norm)
    updates.update(lasagne.updates.adam(grads, all_parameters, 0.00001))  # we maximize fitness
    print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
    iter_train = theano.function([],
                                 []
                                 + [fitness]
                                 #+ [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters]
                                 + theano_to_print
                                 #+ [T.grad(theano_to_print[2], all_parameters[1], return_disconnected='None')]
                                 ,
                                 updates=updates,
                                 )
    iter_test = theano.function([],[fitness])
    with open("theano-function.pkl", 'wb') as f:
        pickle.dump(iter_train, f, pickle.HIGHEST_PROTOCOL)
else:
    print "Loading theano function since %s..." % strftime("%H:%M:%S", localtime())
    with open("theano-function.pkl", 'rb') as f:
        iter_train = pickle.load(f)

#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])

PARAMETERS_FILE = "optimized-parameters.pkl"
def load_parameters():
    with open(PARAMETERS_FILE, 'rb') as f:
        resume_metadata = pickle.load(f)
        lasagne.layers.set_all_param_values(controller["output"], resume_metadata['param_values'])
    return "Finished"

def dump_parameters():
    with open(PARAMETERS_FILE, 'wb') as f:
        pickle.dump({
            'param_values': lasagne.layers.get_all_param_values(controller["output"])
        }, f, pickle.HIGHEST_PROTOCOL)

def are_there_NaNs(result):
    return np.isfinite(result).all() \
        and all([np.isfinite(p).all() for p in lasagne.layers.get_all_param_values(controller["output"])])

if not args.restart:
    print "Loading parameters... ", load_parameters()
else:
    dump_parameters()

print "Running since %s..." % strftime("%H:%M:%S", localtime())
import time
for i in xrange(100):
    t = time.time()
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "train:", time.time()-t

    t = time.time()
    print iter_test(),datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "test:", time.time()-t



print "Finished on %s..." % strftime("%H:%M:%S", localtime())