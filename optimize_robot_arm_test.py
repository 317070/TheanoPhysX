import math
import theano
import theano.tensor as T
from BatchTheanoPhysicsSystem import BatchedTheanoRigid3DBodyEngine
import lasagne
import sys
import numpy as np
from time import strftime, localtime
import cPickle as pickle
import argparse
from custom_ops import mulgrad





EXP_NAME = "exp8-arm"




def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,3+engine.num_sensors+MEMORY_SIZE), name="sensor_values")
    l_1 = lasagne.layers.DenseLayer(l_input, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1 = lasagne.layers.DenseLayer(l_1, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_2 = lasagne.layers.DenseLayer(l_1, engine.num_motors,
                                         nonlinearity=lasagne.nonlinearities.identity,
                                         W=lasagne.init.Constant(0.0),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_result = l_2
    result = {
        "input":l_input,
        "output":l_result,
    }

    if MEMORY_SIZE>0:
        l_recurrent = lasagne.layers.DenseLayer(l_1, MEMORY_SIZE,
                                             nonlinearity=lasagne.nonlinearities.tanh,
                                             W=lasagne.init.Constant(0.0),
                                             b=lasagne.init.Constant(0.0),
                                             )
        result["recurrent"] = l_recurrent

    return result


def build_model():

    def get_shared_variables():
        return controller_parameters + engine.getSharedVariables() + [target]

    def control_loop(state, memory):
        positions, velocities, rot_matrices = state
        #sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        #objective_sensor = T.sum((target - positions[:,grapper_id,:])**2,axis=1)[:,None]
        ALPHA = 0.95
        if "recurrent" in controller:
            network_input = T.concatenate([ target, memory],axis=1)
            controller["input"].input_var = network_input
            memory = lasagne.layers.helper.get_output(controller["recurrent"])
            memory = mulgrad(memory,ALPHA)
        else:
            network_input = T.concatenate([ target],axis=1)
            controller["input"].input_var = network_input
            memory = None

        motor_signals = lasagne.layers.helper.get_output(controller["output"])

        positions, velocities, rot_matrices = mulgrad(positions, ALPHA), mulgrad(velocities, ALPHA), mulgrad(rot_matrices, ALPHA)
        newstate = engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)
        if "recurrent" in controller:
            newstate += (memory,)
        return newstate

    if "recurrent" in controller:
        empty_memory = (np.array([0]*(MEMORY_SIZE*BATCH_SIZE), dtype='float32').reshape((BATCH_SIZE, MEMORY_SIZE)),)
    else:
        empty_memory = ()

    outputs, updates = theano.scan(
        fn=lambda a,b,c,m,*ns: control_loop(state=(a,b,c), memory=m),
        outputs_info=engine.getInitialState() + empty_memory,
        n_steps=int(math.ceil(total_time/engine.DT)),
        strict=True,
        non_sequences=get_shared_variables()
    )
    #print updates
    assert len(updates)==0
    return outputs, controller_parameters, updates



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
jsonfile = "robotmodel/robot_arm.json"
engine.load_robot_model(jsonfile)
grapper_id = engine.getObjectIndex("sphere2")
BATCH_SIZE = 32
MEMORY_SIZE = 0
engine.compile(batch_size=BATCH_SIZE)
print "#sensors:", engine.num_sensors
print "#motors:", engine.num_motors
#engine.randomizeInitialState(rotate_around="spine")

# step 2: build the model, controller and engine for simulation
total_time = 8

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
    return res.astype('float32')


target = theano.shared(sample(), name="target")
target.set_value(sample())

controller = build_controller()
top_layer = lasagne.layers.MergeLayer(
    incomings=[controller[key] for key in controller if key != "input"]
)
controller_parameters = lasagne.layers.helper.get_all_params(top_layer)

import string
print string.ljust("  layer output shapes:",26),
print string.ljust("#params:",10),
print string.ljust("#data:",10),
print "output shape:"
def comma_seperator(v):
    return '{:,.0f}'.format(v)

all_layers = lasagne.layers.get_all_layers(top_layer)
all_params = lasagne.layers.get_all_params(top_layer, trainable=True)
num_params = sum([np.prod(p.get_value().shape) for p in all_params])

for layer in all_layers[:-1]:
    name = string.ljust(layer.__class__.__name__, 22)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(comma_seperator(num_param), 10)
    num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
    print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
print "  number of parameters:", comma_seperator(num_params)

def build_objectives(states_list):
    positions, velocities, rotations = states_list[:3]
    return (target[:,:] - positions[-1,:,grapper_id,:]).norm(L=2,axis=1)


PARAMETERS_FILE = "optimized-parameters-%s.pkl" % EXP_NAME
def load_parameters():
    print "loading parameters"
    with open(PARAMETERS_FILE, 'rb') as f:
        resume_metadata = pickle.load(f)
        lasagne.layers.set_all_param_values(controller["output"], resume_metadata['param_values'])
    return "Finished"

states, all_parameters, updates = build_model()
fitness = build_objectives(states)

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_test = theano.function([],[fitness] + states[:3])
load_parameters()
r = iter_test()
st = r[1:]
print "initial fitness:", r[0]
with open("../PhysXVids/state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
    pickle.dump({
        "states": st,
        "json": open(jsonfile,"rb").read()
    }, f, pickle.HIGHEST_PROTOCOL)
print "Ran test"

for i in xrange(10):
    target.set_value(sample())
    r = iter_test()
    st = r[1:]
    print "fitness:", r[0], np.mean(r[0]), np.sum(r[0]<0.05)*1.0/BATCH_SIZE


print "Finished on %s..." % strftime("%H:%M:%S", localtime())