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

sys.setrecursionlimit(10**6)

print "Started on %s..." % strftime("%H:%M:%S", localtime())
import random
random.seed(0)
np.random.seed(0)

# step 1: load the physics model
engine = BatchedTheanoRigid3DBodyEngine(num_iterations=1)
engine.load_robot_model("robotmodel/full_predator.json")
spine_id = engine.getObjectIndex("spine")
BATCH_SIZE = 1
engine.compile(batch_size=BATCH_SIZE)


# step 2: build the model, controller and engine for simulation
total_time = 5

def build_objectives(states_list):
    positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return (rotations[-1,:,spine_id,:,:] - engine.getInitialState()[2][:,spine_id,:,:]).norm(L=2,axis=(1,2))


def build_controller():
    l_input = lasagne.layers.InputLayer((BATCH_SIZE,81), name="sensor_values")
    l_1 = lasagne.layers.DenseLayer(l_input, 256,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_2 = lasagne.layers.DenseLayer(l_1, 128,
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0),
                                         )
    l_result = lasagne.layers.DenseLayer(l_2, 16,
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

    def control_loop(state, non_sequences):
        positions, velocities, rot_matrices = state
        sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        controller["input"].input_var = sensor_values
        motor_signals = lasagne.layers.helper.get_output(controller["output"])
        return engine.step_from_this_state(state=(positions, velocities, rot_matrices), motor_signals=motor_signals)

    outputs, updates = theano.scan(
        fn=lambda a,b,c,*ns: control_loop(state=(a,b,c),non_sequences=ns),
        outputs_info=engine.getInitialState(),
        n_steps=int(math.ceil(total_time/engine.DT)),
        non_sequences=get_shared_variables(),
        strict=True
    )
    assert len(updates)==0
    return outputs, controller_parameters, updates


controller = build_controller()
controller_parameters = lasagne.layers.helper.get_all_params(controller["output"])

states, all_parameters, updates = build_model()
fitness = build_objectives(states)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)
print "Finding gradient since %s..." % strftime("%H:%M:%S", localtime())
updates.update(lasagne.updates.sgd(-T.mean(fitness), all_parameters, 0.1))  # we maximize fitness

def dump_parameters():
    with open("optimized-parameters.pkl", 'w') as f:
        pickle.dump({
            'param_values': lasagne.layers.get_all_param_values(controller["output"])
        }, f, pickle.HIGHEST_PROTOCOL)
#dump_parameters()

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             []
                             + [fitness]
                             #+ [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters]
                             + theano_to_print
                             #+ [T.grad(theano_to_print[2],all_parameters[1],return_disconnected='None')]
                             ,
                             updates=updates,
                             )
#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])

print "Running since %s..." % strftime("%H:%M:%S", localtime())
for i in xrange(100000):
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")
    dump_parameters()

print "Finished on %s..." % strftime("%H:%M:%S", localtime())