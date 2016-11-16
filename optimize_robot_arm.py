import math
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
from BatchTheanoPhysicsSystem import BatchedTheanoRigid3DBodyEngine
import lasagne
import sys
import numpy as np
from time import strftime, localtime
import datetime
import cPickle as pickle
import argparse
from custom_ops import mulgrad
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

EXP_NAME = "exp13-arm"
PARAMETERS_FILE = "optimized-parameters-%s.pkl" % EXP_NAME

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
BATCH_SIZE = 128
MEMORY_SIZE = 0
engine.compile(batch_size=BATCH_SIZE)
print "#sensors:", engine.num_sensors
print "#motors:", engine.num_motors
#engine.randomizeInitialState(rotate_around="spine")

# step 2: build the model, controller and engine for simulation
total_time = 8

def sample():
    res = np.array([1.]*3*BATCH_SIZE).reshape((BATCH_SIZE,3))
    while (np.sum(res**2, axis=-1) > 1).any() or (np.sum(res**2, axis=-1) < np.sqrt(2)/2).any():
        idx = np.logical_or(np.sqrt(2)/2 > np.sum(res**2, axis=-1),
                            np.sum(res**2, axis=-1) > 1)
        s = np.concatenate([
            np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
            np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
            np.random.uniform(low=0.0, high=1.0, size=(BATCH_SIZE,1))
            ], axis=1).astype('float32')
        res[idx] = s[idx]
    res += np.array([0., 0., 0.1]*BATCH_SIZE, dtype='float32').reshape((BATCH_SIZE,3))
    return res.astype('float32')

target = theano.shared(sample(), name="target")
target.set_value(sample())


def build_objectives_test(states_list):
    positions, velocities, rotations = states_list[:3]
    return T.mean((target[None,:,:] - positions[700:,:,grapper_id,:]).norm(L=2,axis=2),axis=0)

def build_objectives(states_list):
    positions, velocities, rotations = states_list[:3]
    return T.mean((target[None,:,:] - positions[400:,:,grapper_id,:]).norm(L=2,axis=2),axis=0)

def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,3+engine.num_sensors+MEMORY_SIZE), name="sensor_values")
    l_1 = lasagne.layers.DenseLayer(l_input, 1024,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1 = lasagne.layers.dropout(l_1)
    l_1 = lasagne.layers.DenseLayer(l_1, 1024,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1 = lasagne.layers.dropout(l_1)
    l_1 = lasagne.layers.DenseLayer(l_1, 1024,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1 = lasagne.layers.dropout(l_1)
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


def build_model(engine, controller, controller_parameters, deterministic = False):

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
            memory = lasagne.layers.helper.get_output(controller["recurrent"], deterministic = deterministic)
            memory = mulgrad(memory,ALPHA)
        else:
            network_input = T.concatenate([ target],axis=1)
            controller["input"].input_var = network_input
            memory = None

        motor_signals = lasagne.layers.helper.get_output(controller["output"], deterministic = deterministic)

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
    #assert len(updates)==0
    return outputs, updates

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

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
test_engine = BatchedTheanoRigid3DBodyEngine()
jsonfile = "robotmodel/robot_arm.json"
test_engine.load_robot_model(jsonfile)
test_engine.compile(batch_size=BATCH_SIZE)

deterministic_states, det_updates = build_model(test_engine, controller, controller_parameters, deterministic=True)
deterministic_fitness = build_objectives_test(deterministic_states)

iter_test = theano.function([],[deterministic_fitness] + deterministic_states[:3])

if not args.restart:
    load_parameters()

r = iter_test()

st = r[1:]
print "initial fitness:", r[0], np.mean(r[0])
with open("state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
    pickle.dump({
        "states": st,
        "json": open(jsonfile,"rb").read()
    }, f, pickle.HIGHEST_PROTOCOL)
print "Ran test %s..." % strftime("%H:%M:%S", localtime())

states, updates = build_model(engine, controller, controller_parameters, deterministic=True)
fitness = build_objectives(states)
fitness = T.switch(T.isnan(fitness) + T.isinf(fitness), np.float32(0), fitness)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
loss = T.mean(fitness)

grads = theano.grad(loss, controller_parameters)
grads = lasagne.updates.total_norm_constraint(grads, 1.0)

grads = [T.switch(T.isnan(g) + T.isinf(g), np.float32(0), g) for g in grads]


lr = theano.shared(np.float32(0.001))
lr.set_value(np.float32(np.mean(r[0]) / 1000.))
updates.update(lasagne.updates.adam(grads, controller_parameters, lr))  # we maximize fitness

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             [fitness]
                             ,
                             updates=updates,
                             )



print "Running since %s..." % strftime("%H:%M:%S", localtime())
import time
i=0
while True:
    i+=1
    target.set_value(sample())
    st = time.time()
    fitnesses = iter_train()
    print "train fitness:", np.mean(fitnesses), i
    if np.isfinite(fitnesses).all():
        dump_parameters()
    if i%10==0:
        t=sample()
        target.set_value(t)
        st = time.time()
        r = iter_test()
        print "test fitness:", np.mean(r[0])
        lr.set_value(np.float32(np.mean(r[0]) / 1000.))
        with open("state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
            pickle.dump({
                "targets": t,
                "states": r[1:],
                "json": open(jsonfile,"rb").read()
            }, f, pickle.HIGHEST_PROTOCOL)
        if np.mean(r[0])<0.01:
            break



print "Finished on %s..." % strftime("%H:%M:%S", localtime())