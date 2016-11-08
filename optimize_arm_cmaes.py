import math
import theano
import theano.tensor as T
from BatchTheanoPhysicsSystem import BatchedTheanoRigid3DBodyEngine
import lasagne
import sys
import numpy as np
from time import strftime, localtime
import argparse
import cma

# initial: [ 0.42198306,  0.3469744 ,  0.57974786,  0.38210401]
from custom_ops import mulgrad

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
jsonfile = "robotmodel/robot_arm.json"
engine.load_robot_model(jsonfile)
grapper_id = engine.getObjectIndex("sphere2")
BATCH_SIZE = 1
MEMORY_SIZE = 0
engine.compile(batch_size=BATCH_SIZE)
print "#sensors:", engine.num_sensors
print "#motors:", engine.num_motors
#engine.randomizeInitialState(rotate_around="spine")

# step 2: build the model, controller and engine for simulation
total_time = 8

def sample():
    return np.concatenate([
        np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
        np.random.uniform(low=-1.0, high=1.0, size=(BATCH_SIZE,1)),
        np.random.uniform(low=0.0, high=1.0, size=(BATCH_SIZE,1))
        ], axis=1).astype('float32')

target = theano.shared(sample(), name="target")
target.set_value(np.array([[0.5,0.5,0.5]],dtype='float32'))
def build_objectives(states_list):
    positions, velocities, rotations = states_list[:3]
    return T.mean((target[None,:,:] - positions[100:,:,grapper_id,:]).norm(L=2,axis=2),axis=0)

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



controller = build_controller()
top_layer = lasagne.layers.MergeLayer(
    incomings=[controller[key] for key in controller if key != "input"]
)
controller_parameters = lasagne.layers.helper.get_all_params(top_layer)


states, all_parameters, updates = build_model()
fitness = build_objectives(states)

#grads = lasagne.updates.total_norm_constraint(grads, 1.0)

#grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(loss, all_parameters)])+1e-9)
#theano_to_print.append(grad_norm)
print "Compiling since %s..." % strftime("%H:%M:%S", localtime())

iter_test = theano.function([],[fitness])

print "Running since %s..." % strftime("%H:%M:%S", localtime())

options = {'ftarget':1e-2, 'seed':1}
iters = 0

parameters = all_parameters

init_value = np.concatenate([i.get_value().flatten() for i in parameters])

def iter_test_safe(param):
    global iters
    idx = 0
    for p in parameters:
        sh = p.get_value().shape
        p.set_value(param[idx:idx+np.prod(sh)].reshape(sh).astype('float32'))
        idx += np.prod(sh)
    res = iter_test()[0][0]
    iters += 1
    print iters, res
    return res



import time
t = time.time()
print cma.fmin(iter_test_safe, init_value, sigma0=1.0, options=options)
print time.time() - t

print "Finished on %s..." % strftime("%H:%M:%S", localtime())