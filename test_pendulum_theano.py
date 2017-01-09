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
from PhysicsSystem import Rigid3DBodyEngine, EngineState
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
from custom_ops import mulgrad
import time
import matplotlib.pyplot as plt


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
BATCH_SIZE = 1024
engine.compile(batch_size=BATCH_SIZE)

t = time.time()
state = engine.get_state_variables()

result_state = engine.do_time_step(state, motor_signals=np.ones(shape=(BATCH_SIZE,1), dtype='float32'))
image = engine.get_camera_image(result_state,"front_camera")
f = theano.function(list(state),list(result_state))

print "time taken =", time.time() - t
frame = None
t = time.time()

state = [i.get_value() for i in engine.get_initial_state()]

while frame is None or plt.get_fignums():
    t+=engine.DT
    res = f(*state)
    state = EngineState(*res[:3])
    print state.positions[0,engine.get_object_index("top"),:], state.positions[0,engine.get_object_index("sled"),:]
    print "time taken =", time.time() - t
    t = time.time()

print "done"