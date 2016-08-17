import math
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
import lasagne
import sys
import numpy as np
sys.setrecursionlimit(10**6)

# step 1: load the physics model
engine = TheanoRigid3DBodyEngine(num_iterations=10)
engine.load_robot_model("robotmodel/predator.json")
spine_id = engine.getObjectIndex("spine")

engine.compile()


# step 2: build the model, controller and engine for simulation
DT = 0.001
total_time = 0.001

def build_objectives(states_list):
    positions, velocities, rotations = states_list
    return T.sqrt(T.sum((positions[-1,spine_id,:] - engine.getInitialState()[0][spine_id,:])**2, axis=-1))

def build_model():
    controller_parameters = []

    def build_controller(sensor_values):
        l_input = lasagne.layers.InputLayer((1,81), input_var=np.ones(shape=(1,81), dtype='float32'), name="sensor_values")
        #l_input = lasagne.layers.InputLayer((1,81), input_var=sensor_values[None,:], name="sensor_values")
        l_result = lasagne.layers.DenseLayer(l_input, 16,
                                             nonlinearity=lasagne.nonlinearities.identity,
                                             #W=lasagne.init.Constant(0.0),
                                             #b=lasagne.init.Constant(10),
                                             )
        l_out = lasagne.layers.ReshapeLayer(l_result,shape=(-1,))
        return l_out

    def control_loop(state):
        positions, velocities, rot_matrices = state
        sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        controller = build_controller(sensor_values)
        motor_signals = lasagne.layers.helper.get_output(controller)
        controller_parameters[:] = lasagne.layers.helper.get_all_params(controller)
        return engine.step_from_this_state(state=(positions, velocities, rot_matrices), dt=DT, motor_signals=motor_signals)


    outputs, updates = theano.scan(
        fn=lambda a,b,c: control_loop(state=(a,b,c)),
        outputs_info=engine.getInitialState(),
        n_steps=int(math.ceil(total_time/DT))
    )
    assert len(updates)==0
    return outputs, controller_parameters


states, all_parameters = build_model()
fitness = build_objectives(states)

updates = lasagne.updates.sgd(fitness, all_parameters, 0.1)
from time import strftime, localtime
import datetime

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             [fitness, states[0].shape] + [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters],
                             updates=updates,
                             )
#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])
print "Running since %s..." % strftime("%H:%M:%S", localtime())
for i in xrange(10):
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")