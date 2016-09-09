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
sys.setrecursionlimit(10**6)

print "Started on %s..." % strftime("%H:%M:%S", localtime())


# step 1: load the physics model
engine = BatchedTheanoRigid3DBodyEngine(num_iterations=1)
engine.load_robot_model("robotmodel/predator.json")
spine_id = engine.getObjectIndex("spine")
BATCH_SIZE = 1
engine.compile(batch_size=BATCH_SIZE)


# step 2: build the model, controller and engine for simulation
DT = 0.001
total_time = 0.002

def build_objectives(states_list):
    positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,:,6,:,:]])
    return (rotations[-1,:,spine_id,:,:] - engine.getInitialState()[2][:,spine_id,:,:]).norm(L=1,axis=(1,2))

def build_model():
    controller_parameters = []

    def build_controller(sensor_values):
        l_input = lasagne.layers.InputLayer((BATCH_SIZE,81), input_var=sensor_values, name="sensor_values")
        l_result = lasagne.layers.DenseLayer(l_input, 16,
                                             nonlinearity=lasagne.nonlinearities.identity,
                                             W=lasagne.init.Constant(0.0),
                                             b=lasagne.init.Constant(0.01),
                                             )

        return l_result

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
    return outputs, controller_parameters, updates


states, all_parameters, updates = build_model()
fitness = build_objectives(states)

#import theano.printing
#theano.printing.debugprint(T.mean(fitness), print_type=True)

updates.update(lasagne.updates.sgd(-T.mean(fitness), all_parameters, 0.0))  # we maximize fitness


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
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])
print "Running since %s..." % strftime("%H:%M:%S", localtime())
for i in xrange(2):
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")

print "Finished on %s..." % strftime("%H:%M:%S", localtime())