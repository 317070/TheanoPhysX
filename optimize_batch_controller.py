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
from custom_ops import mulgrad

EXP_NAME = "exp7"

parser = argparse.ArgumentParser(description='Process some integers.')
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
jsonfile = "robotmodel/demi_predator.json"
engine.load_robot_model(jsonfile)
spine_id = engine.getObjectIndex("spine")
BATCH_SIZE = 1
engine.compile(batch_size=BATCH_SIZE)
print "#sensors:", engine.num_sensors
#engine.randomizeInitialState(rotate_around="spine")


# step 2: build the model, controller and engine for simulation
total_time = 8

def build_objectives(states_list):
    t, positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return T.mean(velocities[:,:,spine_id,0] * (positions[:,:,spine_id,2]>0),axis=0)
    #return (positions[-1,:,spine_id,:2] - engine.getInitialState()[0][:,spine_id,:2]).norm(L=2,axis=1)


def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,2+engine.num_sensors), name="sensor_values")
    l_1 = lasagne.layers.DenseLayer(l_input, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_2 = lasagne.layers.DenseLayer(l_1, 8,
                                         nonlinearity=lasagne.nonlinearities.identity,
                                         W=lasagne.init.Constant(0.0),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_init = lasagne.layers.DenseLayer(l_input, num_units=8,
                                         nonlinearity=lasagne.nonlinearities.identity,
                                         W=np.array([ 0.8,-0.8,-0.8, 0.8,   0,   0,   0,   0,
                                                        0,   0,   0,   0, 0.5,-0.5,-0.5, 0.5]
                                                    +[0]*(engine.num_sensors*8),dtype='float32').reshape((2+engine.num_sensors, 8)),
                                         b=np.array([ 0.5, 0.5, 0.5, 0.5,   0,   0,   0,   0],dtype='float32'),
                                         )
    l_result = lasagne.layers.ElemwiseSumLayer([l_2, l_init])

    return {
        "input":l_input,
        "output":l_result
    }


def build_model():

    def get_shared_variables():
        return controller_parameters + engine.getSharedVariables()

    def control_loop(state, t):
        positions, velocities, rot_matrices = state
        sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        t = t + engine.DT
        sine = T.tile(T.sin(np.float32(2*np.pi*1.5) * t), BATCH_SIZE)
        cosine = T.tile(T.cos(np.float32(2*np.pi*1.5) * t), BATCH_SIZE)
        sensor_values = T.concatenate([sine[:,None],cosine[:,None], sensor_values],axis=1)
        controller["input"].input_var = sensor_values
        motor_signals = lasagne.layers.helper.get_output(controller["output"])
        ALPHA = 0.95
        positions, velocities, rot_matrices = mulgrad(positions, ALPHA), mulgrad(velocities, ALPHA), mulgrad(rot_matrices, ALPHA)
        return (t,) + engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)

    outputs, updates = theano.scan(
        fn=lambda t,a,b,c,*ns: control_loop(state=(a,b,c), t=t),
        outputs_info=(np.float32(0),)+engine.getInitialState(),
        n_steps=int(math.ceil(total_time/engine.DT)),
        strict=True,
        non_sequences=get_shared_variables(),
    )
    assert len(updates)==0
    return outputs, controller_parameters, updates



controller = build_controller()

controller_parameters = lasagne.layers.helper.get_all_params(controller["output"])

states, all_parameters, updates = build_model()
fitness = build_objectives(states)
fitness = T.switch(T.isnan(fitness) + T.isinf(fitness), np.float32(0), fitness)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
loss = -T.mean(fitness)

grads = theano.grad(loss, all_parameters)
#grads = lasagne.updates.total_norm_constraint(grads, 1.0)

#grads = [T.switch(T.isnan(g) + T.isinf(g), np.float32(0), g) for g in grads]

#grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
#theano_to_print.append(grad_norm)
updates.update(lasagne.updates.adam(grads, all_parameters, 0.000001))  # we maximize fitness
print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_test = theano.function([],[fitness, states[1], states[2], states[3]])
r = iter_test()
st = r[1:]
print "initial fitness:", r[0]
with open("state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
    pickle.dump({
        "states": st,
        "json": open(jsonfile,"rb").read()
    }, f, pickle.HIGHEST_PROTOCOL)
print "Ran test"
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

#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])

PARAMETERS_FILE = "optimized-parameters-%s.pkl" % EXP_NAME
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

print "Running since %s..." % strftime("%H:%M:%S", localtime())
import time
for i in xrange(100000):
    st = time.time()
    fitnesses = iter_train()
    print fitnesses, np.mean(fitnesses), datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "train:", time.time()-st, i
    if np.isfinite(fitnesses).all():
        dump_parameters()
    if i%10==0:
        st = time.time()
        r = iter_test()
        print "test fitness:", r[0]
        with open("state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
            pickle.dump({
                "states": r[1:],
                "json": open(jsonfile,"rb").read()
            }, f, pickle.HIGHEST_PROTOCOL)



print "Finished on %s..." % strftime("%H:%M:%S", localtime())