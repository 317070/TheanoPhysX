import cPickle as pickle
import json
import numpy as np

data = pickle.load(open("../PhysXVids/state-dump-exp13-pendulum.pkl","rb"))
data_json = json.loads(data["json"])
dt = data_json["integration_parameters"]["time_step"]

images = data["images"]
print images.shape

import scipy.misc
scipy.misc.imsave('camera.png', images[15,0,:,:,::-1].transpose(2,1,0))

import matplotlib.pyplot as plt
image = images[0,0]
frame = plt.imshow(image.transpose(2,1,0), interpolation='nearest')
plt.gca().invert_yaxis()
plt.pause(dt)
t = 0
i = 0
while plt.get_fignums():
    i+=1
    t+=dt
    image = images[i%images.shape[0],0]
    frame.set_data(image.transpose(2,1,0))
    plt.title("time=%.3f"%t)
    plt.draw()
    plt.pause(dt)

print "finished"