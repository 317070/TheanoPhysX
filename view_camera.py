import cPickle as pickle
import json
import numpy as np
import time

# data = pickle.load(open("../PhysXVids/state-dump-exp14-pendulum-memory.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp15.pkl","rb"))
data = pickle.load(open("../PhysXVids/state-dump-exp15b.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp16.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp16b.pkl","rb"))
data_json = json.loads(data["json"])
dt = data_json["integration_parameters"]["time_step"]

images = data["images"]


# import scipy.misc
# scipy.misc.imsave('camera.png', images[15,0,:,:,::-1].transpose(2,1,0))

import matplotlib.pyplot as plt
image = images[0,0]
frame = plt.imshow(image.transpose(2,1,0), interpolation='nearest')
plt.gca().invert_yaxis()
plt.pause(dt)
t = 0
i = 0
video_length = images.shape[0]*dt
last_t = time.time()
while plt.get_fignums():
    i+=1
    t= (t - last_t + time.time()) % video_length
    imageR = images[int(t/dt),0,0:1]
    imageG = images[int(t/dt)-1,0,1:2]
    imageB = images[int(t/dt)-10,0,2:3]
    image = np.concatenate([imageR,imageG,imageB],axis=0)
    frame.set_data(image.transpose(2,1,0))
    plt.title("time=%.3f"%t)
    last_t = time.time()
    plt.draw()
    plt.pause(dt)


print "finished"