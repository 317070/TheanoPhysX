import math
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine, theano_to_print
import lasagne
import sys
import numpy as np
sys.setrecursionlimit(10**6)

# step 1: load the physics model
engine = TheanoRigid3DBodyEngine()
engine.load_robot_model("robotmodel/full_predator.json")
spine_id = engine.get_object_index("spine")

engine.compile()


# step 2: build the model, controller and engine for simulation
DT = 0.001
total_time = 0.002

def build_objectives(states_list):
    positions, velocities, rotations = states_list
    #theano_to_print.extend([rotations[-1,6,:,:]])
    return (rotations[-1,spine_id,:,:] - engine.getInitialState()[2][spine_id,:,:]).norm(L=1,axis=(0,1))

def build_model():
    controller_parameters = []

    def build_controller(sensor_values):
        l_input = lasagne.layers.InputLayer((1,81), input_var=np.ones(shape=(1,81), dtype='float32'), name="sensor_values")
        l_1 = lasagne.layers.DenseLayer(l_input, 16,
                                             nonlinearity=lasagne.nonlinearities.identity,
                                             W=lasagne.init.Constant(0.0),
                                             b=lasagne.init.Constant(0.01),
                                             )
        l_result = lasagne.layers.DenseLayer(l_input, 16,
                                             nonlinearity=lasagne.nonlinearities.identity,
                                             W=lasagne.init.Constant(0.0),
                                             b=lasagne.init.Constant(0.01),
                                             )
        l_out = lasagne.layers.ReshapeLayer(l_result,shape=(-1,))
        return l_out

    def control_loop(state):
        positions, velocities, rot_matrices = state
        sensor_values = engine.get_sensor_values(state=(positions, velocities, rot_matrices))
        controller = build_controller(sensor_values)
        motor_signals = lasagne.layers.helper.get_output(controller)
        controller_parameters[:] = lasagne.layers.helper.get_all_params(controller)
        return engine.step_from_this_state(state=(positions, velocities, rot_matrices), dt=DT, motor_signals=motor_signals)


    outputs, updates = theano.scan(
        fn=lambda a,b,c: control_loop(state=(a,b,c)),
        outputs_info=engine.getInitialState(),
        n_steps=int(total_time/engine.DT)
    )
    assert len(updates)==0
    return outputs, controller_parameters, updates


states, all_parameters, updates = build_model()
fitness = build_objectives(states)

updates.update(lasagne.updates.sgd(fitness, all_parameters, 0.0))  # we maximize fitness
from time import strftime, localtime
import datetime

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

iter_test = theano.function([],
                             []
                             + [fitness]
                             #+ [T.max(abs(T.grad(fitness,param,return_disconnected='None'))) for param in all_parameters]
                             #+ [T.grad(theano_to_print[2],all_parameters[1],return_disconnected='None')]
                             )
#print "Running since %s..." % strftime("%H:%M:%S", localtime())
#import theano.printing
#theano.printing.debugprint(iter_train.maker.fgraph.outputs[0])
print "Running since %s..." % strftime("%H:%M:%S", localtime())
import time
for i in xrange(100):
    t = time.time()
    print iter_train(),datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "train:", time.time()-t

    t = time.time()
    print iter_test(),datetime.datetime.now().strftime("%H:%M:%S.%f")
    print "test:", time.time()-t


print "Finished on %s..." % strftime("%H:%M:%S", localtime())