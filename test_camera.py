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

image = engine.getCameraImage()

import matplotlib.pyplot as plt
plt.imshow(image/255., interpolation='nearest')
plt.gca().invert_yaxis()
plt.show()
print "done"