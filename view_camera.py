import cPickle as pickle
import json
import numpy as np
import time
import os


# data = pickle.load(open("../PhysXVids/state-dump-exp14-pendulum-memory.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp15.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp15b.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp16.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp16b.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp16d.pkl","rb"))
# data = pickle.load(open("../PhysXVids/state-dump-exp17.pkl","rb"))
data = pickle.load(open("../PhysXVids/state-dump-exp18.pkl","rb"))
#data_json = json.loads(data["json"])
dt = 0.04#4*data_json["integration_parameters"]["time_step"]

images = data["images"]


# import scipy.misc
# scipy.misc.imsave('camera.png', images[15,0,:,:,::-1].transpose(2,1,0))

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['Computer Modern Roman'])


image = images[0,0]
fig = plt.figure(figsize=(5,2.5))

ax = plt.subplot(111)
frame = ax.imshow(image.transpose(2,1,0), interpolation='nearest')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().invert_yaxis()
plt.axis('off')
plt.figtext(.5,0.085,"Pendulum-cart with only a camera sensor. Solution by a deep neural network", fontsize=10, ha='center')
plt.figtext(.5,0.025,"optimized with a differentiable camera in a differentiable physics engine", fontsize=10, ha='center')
plt.pause(dt)
t = 0
i = 0
video_length = images.shape[0]*dt
last_t = time.time()

frame_dt = 1./25.

live = False
if not live:
    import shutil
    try:
        shutil.rmtree('gifs')
    except OSError:
        pass
    os.makedirs('gifs')


def on_close(event):
    event.canvas.figure.has_been_closed = True
    print 'Closed Figure'
fig.canvas.mpl_connect('close_event', on_close)
fig.has_been_closed = False
while not fig.has_been_closed:
    imageR = images[int(t/dt),0,0:1]
    imageG = images[int(t/dt)-0,0,1:2]
    imageB = images[int(t/dt)-0,0,2:3]
    image = np.concatenate([imageR,imageG,imageB],axis=0)
    frame.set_data(image.transpose(2,1,0))
    text = plt.figtext(.5,0.9,"$\\left(t=%.2f\,\mathrm{s}\\right)$"%t, fontsize=12, ha='center')

    last_t = time.time()
    plt.draw()
    plt.pause(dt)
    if not live:
        #plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig('gifs/image%05d.png'%i)

    text.remove()

    i+=1
    if live:
        t = (t - last_t + time.time()) % video_length
    else:
        t = t + frame_dt
        if t>video_length:
            plt.close('all')

if not live:
    os.system("convert -coalesce -layers Optimize -delay 4 -loop 0 gifs/*.png gifs/animated.gif")
    print "Made animated gif"

print "finished"