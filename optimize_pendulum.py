import math
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
import lasagne
import sys
import numpy as np
from time import strftime, localtime
import datetime
import cPickle as pickle
import argparse
from PhysicsSystem import Z, EngineState
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
from custom_ops import mulgrad
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

EXP_NAME = "exp13-pendulum"
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
engine = TheanoRigid3DBodyEngine()
jsonfile = "robotmodel/pendulum.json"
engine.load_robot_model(jsonfile)
top_id = engine.get_object_index("top")
total_time = 10
BATCH_SIZE = 1
MEMORY_SIZE = 128

CAMERA = "front_camera"
engine.compile(batch_size=BATCH_SIZE)
print "#batch:", BATCH_SIZE
print "#memory:", MEMORY_SIZE
print "#sensors:", engine.num_sensors
print "#motors:", engine.num_motors
print "#cameras:", engine.num_cameras
#engine.randomizeInitialState(rotate_around="spine")

# step 2: build the model, controller and engine for simulation


def build_objectives(states_list):
    positions, velocities, rotations = states_list[:3]
    return T.mean(positions[:,:,top_id,Z],axis=0)-0.3


def build_objectives_test(states_list):
    positions, velocities, rotations = states_list[:3]
    return T.mean(positions[700:,:,top_id,Z],axis=0)-0.3


def build_controller():
    l_input = lasagne.layers.InputLayer(engine.get_camera_image_size(CAMERA), name="image_inputs")

    l_memory = lasagne.layers.InputLayer((BATCH_SIZE,MEMORY_SIZE), name="memory_values")

    l_1a = lasagne.layers.Conv2DLayer(l_input, 32, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1b = lasagne.layers.Conv2DLayer(l_1a, 32, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_1 = lasagne.layers.MaxPool2DLayer(l_1b, pool_size=(2,2))

    l_2a = lasagne.layers.Conv2DLayer(l_1, 64, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_2b = lasagne.layers.Conv2DLayer(l_2a, 64, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_2 = lasagne.layers.MaxPool2DLayer(l_2b, pool_size=(2,2))


    l_3a = lasagne.layers.Conv2DLayer(l_2, 32, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_3b = lasagne.layers.Conv2DLayer(l_3a, 32, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_3 = lasagne.layers.MaxPool2DLayer(l_3b, pool_size=(2,2))


    l_4a = lasagne.layers.Conv2DLayer(l_3, 16, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_4b = lasagne.layers.Conv2DLayer(l_4a, 16, filter_size=(3,3),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_4 = lasagne.layers.MaxPool2DLayer(l_4b, pool_size=(2,2))

    l_5 = lasagne.layers.DenseLayer(l_4, MEMORY_SIZE,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )


    l_flat = lasagne.layers.ConcatLayer([lasagne.layers.flatten(l_5),
                                     lasagne.layers.flatten(l_memory)])

    l_d1 = lasagne.layers.DenseLayer(l_flat, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )

    l_d1 = lasagne.layers.DenseLayer(l_d1, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )

    l_d = lasagne.layers.DenseLayer(l_d1, engine.num_motors,
                                         nonlinearity=lasagne.nonlinearities.tanh,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_result = l_d
    result = {
        "input":l_input,
        "output":l_result,
    }

    if MEMORY_SIZE>0:
        l_recurrent = lasagne.layers.DenseLayer(l_d1, MEMORY_SIZE,
                                             nonlinearity=lasagne.nonlinearities.tanh,
                                             W=lasagne.init.Orthogonal(),
                                             b=lasagne.init.Constant(0.0),
                                             )
        result["recurrent"] = l_recurrent
        result["memory"] = l_memory

    return result


def build_model(engine, controller, controller_parameters, deterministic = False):

    def get_shared_variables():
        return controller_parameters + engine.get_shared_variables()

    def control_loop(state, memory):
        positions, velocities, rot_matrices = state
        #sensor_values = engine.get_sensor_values(state=(positions, velocities, rot_matrices))
        ALPHA = 1.0
        image = engine.get_camera_image(EngineState(*state),CAMERA)
        controller["input"].input_var = image
        if "recurrent" in controller:
            controller["memory"].input_var = memory
            memory = lasagne.layers.helper.get_output(controller["recurrent"], deterministic = deterministic)
            memory = mulgrad(memory,ALPHA)

        motor_signals = lasagne.layers.helper.get_output(controller["output"], deterministic = deterministic)

        positions, velocities, rot_matrices = mulgrad(positions, ALPHA), mulgrad(velocities, ALPHA), mulgrad(rot_matrices, ALPHA)
        newstate = engine.do_time_step(state=EngineState(positions, velocities, rot_matrices), motor_signals=motor_signals)
        if "recurrent" in controller:
            newstate += (memory,)
        return newstate

    if "recurrent" in controller:
        empty_memory = (np.array([0]*(MEMORY_SIZE*BATCH_SIZE), dtype='float32').reshape((BATCH_SIZE, MEMORY_SIZE)),)
    else:
        empty_memory = ()

    outputs, updates = theano.scan(
        fn=lambda a,b,c,m,*ns: control_loop(state=(a,b,c), memory=m),
        outputs_info=engine.get_initial_state() + empty_memory,
        n_steps=int(math.ceil(total_time/engine.DT)),
        strict=True,
        non_sequences=get_shared_variables()
    )
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
test_engine = TheanoRigid3DBodyEngine()
test_engine.load_robot_model(jsonfile)
test_engine.compile(batch_size=BATCH_SIZE)

deterministic_states, det_updates = build_model(test_engine, controller, controller_parameters, deterministic=True)
deterministic_fitness = build_objectives(deterministic_states)

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
#fitness = T.switch(T.isnan(fitness) + T.isinf(fitness), np.float32(0.001), fitness)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
# we want to maximize fitness
loss = -T.mean(fitness)

grads = theano.grad(loss, controller_parameters,
                    disconnected_inputs="warn",
                             return_disconnected="zero")
grads = lasagne.updates.total_norm_constraint(grads, 1.0)

#grads = [T.switch(T.isnan(g) + T.isinf(g), np.float32(0.0), g) for g in grads]


lr = theano.shared(np.float32(0.001))
#lr.set_value(np.float32(np.mean(r[0]) / 1000.))
updates.update(lasagne.updates.adam(grads, controller_parameters, lr))  # we maximize fitness

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             [fitness]
                             ,
                             updates=updates
                             )



print "Running since %s..." % strftime("%H:%M:%S", localtime())
import time
i=0
while True:
    i+=1
    st = time.time()
    fitnesses = iter_train()
    print fitnesses, np.mean(fitnesses), datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "train:", time.time()-st, i
    if np.isfinite(fitnesses).all():
        dump_parameters()
    if i%10==0:
        st = time.time()
        r = iter_test()
        print "test fitness:", r[0], np.mean(r[0])
        #lr.set_value(np.float32(np.mean(r[0]) / 1000.))
        with open("state-dump-%s.pkl"%EXP_NAME, 'wb') as f:
            pickle.dump({
                "states": r[1:],
                "json": open(jsonfile,"rb").read()
            }, f, pickle.HIGHEST_PROTOCOL)
        if np.mean(r[0])>0.55:
            break



print "Finished on %s..." % strftime("%H:%M:%S", localtime())