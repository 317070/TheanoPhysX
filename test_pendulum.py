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
from PhysicsSystem import Rigid3DBodyEngine
from custom_ops import mulgrad
import time


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
engine = Rigid3DBodyEngine()
jsonfile = "robotmodel/abstract_art.json"
engine.load_robot_model(jsonfile)
BATCH_SIZE = 1
engine.compile()

t = time.time()
state = engine.get_initial_state()
image = engine.get_camera_image(state,"front_camera")
print "time taken =", time.time() - t

import matplotlib.pyplot as plt
frame = plt.imshow(image, interpolation='nearest')
plt.gca().invert_yaxis()
plt.pause(engine.DT)
t = 0
while plt.get_fignums():
    t+=engine.DT
    print t
    state = engine.do_time_step(state,dt=engine.DT, motor_signals=[2*np.sin(t)])
    image = engine.get_camera_image(state,"front_camera")
    frame.set_data(image)
    plt.draw()
    plt.pause(engine.DT)

print "done"