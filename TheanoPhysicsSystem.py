from collections import namedtuple
import json
from math import pi
from PhysicsSystem import Rigid3DBodyEngine

__author__ = 'jonas'
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.seterr(all='raise')
eps=1e-4

X = 0
Y = 1
Z = 2



EngineState = namedtuple(typename='EngineState', field_names=["positions", "velocities", "rotations"])


class TheanoRigid3DBodyEngine(Rigid3DBodyEngine):
    def __init__(self, *args, **kwargs):
        super(TheanoRigid3DBodyEngine, self).__init__(*args, **kwargs)

    def compile(self, *args,**kwargs):
        super(TheanoRigid3DBodyEngine,self).compile(*args,**kwargs)

    def getSharedVariables(self):
        return [
            self.positionVectors,
            self.velocityVectors,
            self.rot_matrices,
            self.lower_inertia_inv,
            self.upper_inertia_inv,
        ]

    def getCameraImage(self, camera_name, *args, **kwargs):
        super(TheanoRigid3DBodyEngine,self).getCameraImage(camera_name, *args,**kwargs)

    def evaluate(self, *args, **kwargs):
        pass

    def do_time_step(self, state=None, dt=None, motor_signals=list()):
        if dt is None:
            dt = self.DT

        positions, velocities, rot_matrices = state
        ##################
        # --- Step 3 --- #
        ##################
        # semi-implicit Euler integration
        velocities = self.evaluate(dt, positions, velocities, rot_matrices, motor_signals=motor_signals)

        positions = positions + velocities[:,:,:3] * dt
        # TODO: batch-dot-product
        rot_matrices = self.normalize_matrix(rot_matrices[:,:,:,:] + T.sum(rot_matrices[:,:,:,:,None] * batch_skew_symmetric(dt * velocities[:,:,3:])[:,:,None,:,:],axis=3) )

        return positions, velocities, rot_matrices

    def getInitialState(self):
        return EngineState(positions=self.positionVectors,
                           velocities=self.velocityVectors,
                           rotations=self.rot_matrices)

    def getInitialPosition(self, reference):
        idx = self.getObjectIndex(reference)
        return self.positionVectors[:,idx,:]

    def getInitialRotation(self, reference):
        idx = self.getObjectIndex(reference)
        return self.rot_matrices[:,idx,:,:]

    def getObjectIndex(self, reference):
        return self.objects[reference]

    def getSensorValues(self, state):
        # make positionvectors neutral according to reference object
        positions, velocities, rot_matrices = state

        r = []
        for sensor in self.sensors:
            idx = self.getObjectIndex(sensor["object"])
            if "reference" in sensor:
                ref_idx = self.getObjectIndex(sensor["reference"])
            else:
                ref_idx = None
            if sensor["type"] == "position":
                if ref_idx:
                    res = positions[:,idx,:]-positions[:,ref_idx,:]
                    rot = rot_matrices[:,ref_idx,:,:]
                    axis = sensor["axis"]
                    axis = theano_convert_model_to_world_coordinate_no_bias(axis[None,:], rot)
                else:
                    res = positions[:,idx,:]
                    axis = sensor["axis"]
                r.append(T.sum(res*axis,axis=1))

            if sensor["type"] == "velocity":
                if ref_idx:
                    res = velocities[:,idx,:3]-velocities[:,ref_idx,:3]
                    rot = rot_matrices[:,ref_idx,:,:]
                    axis = sensor["axis"]
                    axis = theano_convert_model_to_world_coordinate_no_bias(axis[None,:], rot)
                else:
                    res = velocities[:,idx,:3]
                    axis = sensor["axis"]
                r.append(T.sum(res*axis,axis=1))

            if sensor["type"] == "orientation":
                if ref_idx:
                    res = T.sum(rot_matrices[:,idx,:,:,None]*rot_matrices[:,ref_idx,None,:,:], axis=3)
                    rot = rot_matrices[:,ref_idx,:,:]
                    axis = sensor["axis"][None,:]
                    axis = theano_convert_model_to_world_coordinate_no_bias(axis, rot)
                else:
                    res = rot_matrices[:,idx,:,:]
                    axis = sensor["axis"][None,:]

                # if angle is >90degrees, you're in the negative lobe
                #cos_theta = (res[0,0] + res[1,1] + res[2,2] - 1)/2
                #sign = ((cos_theta>0) * 2 - 1)

                # gimbal lock can occur witht this sensor
                r.append(T.sum(axis[:,:,None]*res[:,:,:]*axis[:,None,:], axis=(1,2)))

        result = theano_stack_batched_integers_mixed_numpy(r)

        return result
