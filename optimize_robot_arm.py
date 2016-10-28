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

EXP_NAME = "exp8"

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
BATCH_SIZE = 4
MEMORY_SIZE = 16
engine.compile(batch_size=BATCH_SIZE)
print "#sensors:", engine.num_sensors
engine.randomizeInitialState(rotate_around="spine")


# step 2: build the model, controller and engine for simulation
total_time = 8

def get_target():
    srng = RandomStreams(seed=317070)
    d_x = srng.uniform(size=(BATCH_SIZE,1), low=-1.5, high=1.5)
    d_y = srng.uniform(size=(BATCH_SIZE,1), low=-1.5, high=1.5)
    d_z = srng.uniform(size=(BATCH_SIZE,1), low=0.0, high=1.5)
    return T.concatenate([d_x, d_y, d_z],axis=1), [i[0] for i in srng.updates()]

target, shared_target_vars = get_target()
print shared_target_vars
def build_objectives(states_list):
    t, positions, velocities, rotations = states_list
    return T.mean(T.sum((target[None,:,:] - positions[:,:,grapper_id,:])**2,axis=2),axis=0)

def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,1+engine.num_sensors+MEMORY_SIZE), name="sensor_values")
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
    l_2 = lasagne.layers.DenseLayer(l_1, 4,
                                         nonlinearity=lasagne.nonlinearities.identity,
                                         W=lasagne.init.Constant(0.0),
                                         b=lasagne.init.Constant(0.0),
                                         )

    l_recurrent = lasagne.layers.DenseLayer(l_1, MEMORY_SIZE,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Constant(0.0),
                                         b=lasagne.init.Constant(0.0),
                                         )

    l_result = l_2

    return {
        "input":l_input,
        "output":l_result,
        "recurrent":l_recurrent
    }


def build_model():

    def get_shared_variables():
        return controller_parameters + engine.getSharedVariables() + shared_target_vars

    def control_loop(state, memory):
        positions, velocities, rot_matrices = state
        #sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        objective_sensor = T.sum((target - positions[:,grapper_id,:])**2,axis=1)[:,None]
        network_input = T.concatenate([objective_sensor, memory],axis=1)
        controller["input"].input_var = network_input
        motor_signals = lasagne.layers.helper.get_output(controller["output"])
        memory = lasagne.layers.helper.get_output(controller["recurrent"])

        ALPHA = 0.95
        positions, velocities, rot_matrices = mulgrad(positions, ALPHA), mulgrad(velocities, ALPHA), mulgrad(rot_matrices, ALPHA)
        memory = mulgrad(memory,ALPHA)

        return (memory,) + engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)

    empty_memory = np.array([0]*(MEMORY_SIZE*BATCH_SIZE), dtype='float32').reshape((BATCH_SIZE, MEMORY_SIZE))

    outputs, updates = theano.scan(
        fn=lambda m,a,b,c,*ns: control_loop(state=(a,b,c), memory=m),
        outputs_info=(empty_memory,)+engine.getInitialState(),
        n_steps=int(math.ceil(total_time/engine.DT)),
        strict=True,
        non_sequences=get_shared_variables(),
    )
    print updates
    assert len(updates)==0
    return outputs, controller_parameters, updates



controller = build_controller()
top_layer = lasagne.layers.MergeLayer(
    incomings=[controller["output"], controller["recurrent"]]
)
controller_parameters = lasagne.layers.helper.get_all_params(top_layer)

import string
print string.ljust("  layer output shapes:",26),
print string.ljust("#params:",10),
print string.ljust("#data:",10),
print "output shape:"
def comma_seperator(v):
    return '{:,.0f}'.format(v)

all_layers = lasagne.layers.get_all_layers(controller["output"])
all_params = lasagne.layers.get_all_params(controller["output"], trainable=True)
num_params = sum([np.prod(p.get_value().shape) for p in all_params])

for layer in all_layers[:-1]:
    name = string.ljust(layer.__class__.__name__, 22)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(comma_seperator(num_param), 10)
    num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
    print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
print "  number of parameters:", comma_seperator(num_params)



states, all_parameters, updates = build_model()
fitness = build_objectives(states)
fitness = T.switch(T.isnan(fitness) + T.isinf(fitness), np.float32(0), fitness)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
loss = -T.mean(fitness)

grads = theano.grad(loss, all_parameters)
grads = lasagne.updates.total_norm_constraint(grads, 1.0)

grads = [T.switch(T.isnan(g) + T.isinf(g), np.float32(0), g) for g in grads]

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
                             #+ theano_to_print
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