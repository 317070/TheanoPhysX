import math
import theano
import theano.tensor as T
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
import lasagne
import sys
sys.setrecursionlimit(10**6)

# step 1: load the physics model
engine = TheanoRigid3DBodyEngine()
engine.load_robot_model("robotmodel/predator.json")
engine.compile()


# step 2: build the model, controller and engine for simulation
DT = 1./6000.
total_time = 2

def build_controller(sensor_values):
    l_input = lasagne.layers.InputLayer((1,81), input_var=sensor_values[None,:], name="sensor_values")
    l_result = lasagne.layers.DenseLayer(l_input, 16, nonlinearity=lasagne.nonlinearities.identity)
    l_out = lasagne.layers.ReshapeLayer(l_result,shape=(-1,))
    return l_out


def build_objectives(states_list):
    positions, velocities, rotations = states_list
    return T.mean(T.sum(positions[-1,:,:]**2, axis=-1), axis=-1)


def build_model():
    controller_parameters = []

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

    return outputs, controller_parameters


states, all_parameters = build_model()
fitness = build_objectives(states)

updates = lasagne.updates.sgd(fitness, all_parameters, 0.001)
from time import strftime, localtime
import datetime

print "Compiling since %s..." % strftime("%H:%M:%S", localtime())
iter_train = theano.function([],
                             [fitness],
                             updates=updates
                             )
print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])
print "Running since %s..." % strftime("%H:%M:%S", localtime())
while True:
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")