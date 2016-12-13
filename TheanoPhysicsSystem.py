import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import namedtuple
import json
from math import pi
from PhysicsSystem import Rigid3DBodyEngine, EngineState
from aux import numpy_repeat_new_axis
from aux_theano import batch_skew_symmetric, theano_convert_model_to_world_coordinate_no_bias, \
    theano_stack_batched_integers_mixed_numpy, single_skew_symmetric, theano_dot_last_dimension_vector_matrix, \
    theano_dot_last_dimension_vectors, theano_dot_last_dimension_matrices

__author__ = 'jonas degrave'

#np.seterr(all='raise')
eps=1e-4
X = 0
Y = 1
Z = 2


class TheanoRigid3DBodyEngine(Rigid3DBodyEngine):
    def __init__(self, *args, **kwargs):
        super(TheanoRigid3DBodyEngine, self).__init__(*args, **kwargs)
        self.batch_size = None
        self.initial_positions = None
        self.initial_velocities = None
        self.initial_rotations = None
        self.lower_inertia_inv = None
        self.upper_inertia_inv = None
        self.impulses_P = None

    def compile(self, batch_size, *args,**kwargs):
        super(TheanoRigid3DBodyEngine, self).compile(*args,**kwargs)

        self.batch_size = batch_size
        self.initial_positions = theano.shared(numpy_repeat_new_axis(self.positionVectors, self.batch_size).astype('float32'), name="initial_positions")
        self.initial_velocities = theano.shared(numpy_repeat_new_axis(self.velocityVectors, self.batch_size).astype('float32'), name="initial_velocities")
        self.initial_rotations = theano.shared(numpy_repeat_new_axis(self.rot_matrices, self.batch_size).astype('float32'), name="initial_rotations", )
        self.lower_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,:3,:3]), self.batch_size).astype('float32'), name="lower_inertia_inv", )
        self.upper_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,3:,3:]), self.batch_size).astype('float32'), name="upper_inertia_inv", )
        # For the warm start, keep the impuls of the previous timestep
        self.impulses_P = theano.shared(T.zeros((self.batch_size, self.num_constraints,)), name="impulses_P")


    def getAllSharedVariables(self):
        return [
            self.initial_positions,
            self.initial_velocities,
            self.initial_rotations,
            self.lower_inertia_inv,
            self.upper_inertia_inv,
            self.impulses_P,
        ]

    def doTimeStep(self, state=None, dt=None, motor_signals=list()):
        if dt is None:
            dt = self.DT

        positions, velocities, rotations = state.positions, state.velocities, state.rotations
        ##################
        # --- Step 3 --- #
        ##################
        # semi-implicit Euler integration
        velocities = self.evaluate(dt, positions, velocities, rotations, motor_signals=motor_signals)

        positions = positions + velocities[:,:,:3] * dt
        # TODO: batch-dot-product
        rotations = self.normalize_matrix(rotations[:,:,:,:] + T.sum(rotations[:,:,:,:,None] * batch_skew_symmetric(dt * velocities[:,:,3:])[:,:,None,:,:],axis=3) )

        return EngineState(
            positions=positions, 
            velocities=velocities, 
            rotations=rotations)

    def getInitialState(self):
        return EngineState(positions=self.initial_positions,
                           velocities=self.initial_velocities,
                           rotations=self.initial_rotations)

    def getInitialPosition(self, reference):
        idx = self.getObjectIndex(reference)
        return self.initial_positions[:,idx,:]

    def getInitialRotation(self, reference):
        idx = self.getObjectIndex(reference)
        return self.initial_rotations[:,idx,:,:]

    def getObjectIndex(self, reference):
        return self.objects[reference]

    def getSensorValues(self, state):
        # make positionvectors neutral according to reference object
        positions, velocities, rot_matrices = state.positions, state.velocities, state.rotations

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

                # gimbal lock can occur with this sensor
                r.append(T.sum(axis[:,:,None]*res[:,:,:]*axis[:,None,:], axis=(1,2)))

        result = theano_stack_batched_integers_mixed_numpy(r)
        return result

    def getCameraImage(self, camera_name, *args, **kwargs):
        pass

    def evaluate(self, dt, state, motor_signals=[]):
        positions, velocities, rotations = state.positions, state.velocities, state.positions
        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!
        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities newv that tend to violate the constraints.

        # TODO: integrate other forces
        totalforce = np.array([0,0,0,0,0,0], dtype='float32')  # total force acting on body outside of constraints

        acceleration = np.array([0,0,-9.81,0,0,0], dtype='float32')  # acceleration of the default frame
        newv = velocities + dt * acceleration[None,None,:]
        originalv = newv

        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        
        # convert mass matrices to world coordinates
        # changes every timestep
        M = T.zeros(shape=(self.batch_size,self.num_bodies,6,6))
        M00 = self.lower_inertia_inv
        M01 = T.zeros(shape=(self.batch_size,self.num_bodies,3,3))
        M10 = M01
        M11 = T.sum(rotations[:,:,:,None,:,None] * self.upper_inertia_inv[:,:,:,:,None,None] * rotations[:,:,None,:,None,:], axis=(2,3))
        M0 = T.concatenate([M00,M01],axis=3)
        M1 = T.concatenate([M10,M11],axis=3)
        M = T.concatenate([M0,M1],axis=2)

        # constraints are first dimension! We will need to stack them afterwards!
        J = [np.zeros((self.batch_size,6), dtype="float32") for _ in xrange(2 * self.num_constraints)]   # 0 constraints x 0 bodies x 2 objects x 6 velocities
        b_res =   [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies
        b_error = [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies
        C =       [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            idx2 = references[1]
            follows_Newtons_third_law = (idx1 is not None) and (idx2 is not None)

            # If connected to universe, idx2 is None
            if idx2 is None:
                idx1,idx2 = idx1,idx1
            if idx1 is None:
                idx1,idx2 = idx2,idx2

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                r1x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"][None,:], rotations[:,idx1,:,:])
                r2x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"][None,:], rotations[:,idx2,:,:])
                ss_r1x = single_skew_symmetric(r1x)
                ss_r2x = single_skew_symmetric(r2x)
                batched_eye = numpy_repeat_new_axis(np.eye(3, dtype='float32'), self.batch_size)
                complete_J1 = T.concatenate([-batched_eye, ss_r1x],axis=1)
                complete_J2 = T.concatenate([ batched_eye,-ss_r2x],axis=1)
                error = positions[:,idx2,:]+r2x-positions[:,idx1,:]-r1x
                for i in xrange(3):
                    J[2*(c_idx+i)+0] = complete_J1[:,:,i]
                    J[2*(c_idx+i)+1] = complete_J2[:,:,i]

                    b_error[c_idx+i] = error[:,i]
                c_idx += 3

            if constraint== "parallel" or constraint== "plane" or constraint == "slider" or constraint == "fixed":
                batched_eye = numpy_repeat_new_axis(np.eye(3, dtype='float32'), self.batch_size)
                batched_zeros = numpy_repeat_new_axis(np.zeros((3,3), dtype='float32'), self.batch_size)

                complete_J1 = np.concatenate([batched_zeros,-batched_eye],axis=1)
                complete_J2 = np.concatenate([batched_zeros, batched_eye],axis=1)

                for i in xrange(3):
                    J[2*(c_idx+i)+0] = complete_J1[:,:,i]
                    J[2*(c_idx+i)+1] = complete_J2[:,:,i]

                #TODO: this is np?
                #rot_current = theano_dot_last_dimension_matrices(rotations[:,idx2,:,:], rotations[:,idx1,:,:].dimshuffle(0,2,1))
                #rot_diff = np.dot(rot_current, parameters['rot_init'].T)
                #cross = rot_diff.T - rot_diff

                # TODO: add stabilization of this constraint
                b_error[c_idx] = np.zeros(shape=(self.batch_size,))  #cross[1,2]
                b_error[c_idx+1] = np.zeros(shape=(self.batch_size,))  #cross[2,0]
                b_error[c_idx+2] = np.zeros(shape=(self.batch_size,))  #cross[0,1]
                c_idx += 3


            if constraint == "plane" or constraint == "slider":
                if follows_Newtons_third_law:
                    n1 = theano_convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], rotations[:,idx1,:,:])
                else:
                    n1 = parameters['axis1_in_model1_coordinates']
                    n1 = numpy_repeat_new_axis(n1, self.batch_size)
                
                complete_J1 = T.concatenate([-n1 , T.zeros(shape=(self.batch_size,3))],axis=1)
                complete_J2 = -complete_J1
                
                if follows_Newtons_third_law:
                    J[2*c_idx+0] = complete_J1
                J[2*c_idx+1] = complete_J2
                
                if follows_Newtons_third_law:
                    orig_error = theano_convert_model_to_world_coordinate_no_bias(parameters['trans_init_in_model2'],
                                                                           rotations[:,idx2,:,:])
                    pos_error = positions[:,idx2,:] - positions[:,idx1,:] - orig_error
                else:
                    orig_error = parameters['trans_init']
                    orig_error = numpy_repeat_new_axis(orig_error, self.batch_size)
                    pos_error = positions[:,idx2,:] - orig_error

                b_error[c_idx] = theano_dot_last_dimension_vectors(pos_error, n1)

                c_idx += 1


            if constraint == "slider":
                if follows_Newtons_third_law:
                    n2 = theano_convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], rotations[:,idx1,:,:])
                else:
                    n2 = parameters['axis2_in_model1_coordinates']
                    n2 = numpy_repeat_new_axis(n2, self.batch_size)
                
                complete_J1 = T.concatenate([-n2 , T.zeros(shape=(self.batch_size,3))],axis=1)
                complete_J2 = -complete_J1
                
                if follows_Newtons_third_law:
                    J[2*c_idx+0] = complete_J1
                J[2*c_idx+1] = complete_J2
                
                if follows_Newtons_third_law:
                    orig_error = theano_convert_model_to_world_coordinate_no_bias(parameters['trans_init_in_model2'],
                                                                           rotations[:,idx2,:,:])
                    pos_error = positions[:,idx2,:] - positions[:,idx1,:] - orig_error
                else:
                    orig_error = parameters['trans_init']
                    orig_error = numpy_repeat_new_axis(orig_error, self.batch_size)
                    pos_error = positions[:,idx2,:] - orig_error

                b_error[c_idx] = theano_dot_last_dimension_vectors(pos_error, n2)

                c_idx += 1


            if constraint == "hinge":
                a2x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'][None,:], rotations[:,idx2,:,:])
                b1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'][None,:], rotations[:,idx1,:,:])
                c1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'][None,:], rotations[:,idx1,:,:])
                ss_a2x = single_skew_symmetric(a2x)

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)

                J[2*(c_idx+0)+0] = T.concatenate([batched_zeros,-theano_dot_last_dimension_vector_matrix(b1x,ss_a2x)],axis=1)
                J[2*(c_idx+0)+1] = T.concatenate([batched_zeros, theano_dot_last_dimension_vector_matrix(b1x,ss_a2x)],axis=1)
                J[2*(c_idx+1)+0] = T.concatenate([batched_zeros,-theano_dot_last_dimension_vector_matrix(c1x,ss_a2x)],axis=1)
                J[2*(c_idx+1)+1] = T.concatenate([batched_zeros, theano_dot_last_dimension_vector_matrix(c1x,ss_a2x)],axis=1)

                b_error[c_idx+0] = theano_dot_last_dimension_vectors(a2x,b1x)
                b_error[c_idx+1] = theano_dot_last_dimension_vectors(a2x,c1x)
                c_idx += 2

            if constraint == "angular motor":
                ac = parameters['axis_in_model2_coordinates'][None,:]
                a = theano_convert_model_to_world_coordinate_no_bias(ac, rotations[:,idx2,:,:])

                # TODO: remove dimshuffle(0,2,1) by using batched_dot
                rot_current = theano_dot_last_dimension_matrices(rotations[:,idx2,:,:], rotations[:,idx1,:,:].dimshuffle(0,2,1))
                rot_init = numpy_repeat_new_axis(parameters['rot_init'].T, self.batch_size)
                rot_diff = theano_dot_last_dimension_matrices(rot_current, rot_init)

                traces = rot_diff[:,0,0] + rot_diff[:,1,1] + rot_diff[:,2,2]

                # grad when x=-1 or x=1 does not exist for arccos
                theta2 = T.arccos(T.clip(0.5*(traces-1),-1+eps,1-eps))
                cross = rot_diff.dimshuffle(0,2,1) - rot_diff
                dot2 = cross[:,1,2] * ac[:,0] + cross[:,2,0] * ac[:,1] + cross[:,0,1] * ac[:,2]

                theta = ((dot2>0) * 2 - 1) * theta2

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)
                J[2*c_idx+0] = T.concatenate([batched_zeros,-a],axis=1)
                J[2*c_idx+1] = T.concatenate([batched_zeros, a],axis=1)

                motor_signal = motor_signals[:,parameters["motor_id"]]

                if "min" in parameters and "max" in parameters:
                    motor_min = (parameters["min"]/180. * np.pi)
                    motor_max = (parameters["max"]/180. * np.pi)
                    motor_signal = T.clip(motor_signal, motor_min, motor_max).astype('float32')

                def smallestSignedAngleBetween(x, y):
                    a1 = (x - y) % np.float32(2*np.pi)
                    b1 = (y - x) % np.float32(2*np.pi)
                    return T.minimum(a1,b1)*((a1>b1)*2-1)

                error_signal = -smallestSignedAngleBetween(theta, motor_signal)

                if parameters["type"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["type"] == "position":
                    if "delta" in parameters  and "motor_velocity" in parameters:
                        velocity = parameters["motor_velocity"] / 180. * np.pi
                        b_error[c_idx] = dt * T.clip((abs(error_signal) > parameters["delta"]) * error_signal * parameters["motor_gain"], -velocity, velocity)
                    else:
                        b_error[c_idx] = dt * error_signal * parameters["motor_gain"]

                c_idx += 1


            if constraint == "linear motor":

                ac = parameters['axis_in_model2_coordinates']
                a = theano_convert_model_to_world_coordinate_no_bias(ac, rotations[:,idx2,:,:])

                # WORKPOINT
                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)
                
                if follows_Newtons_third_law:
                    J[2*c_idx+0] = T.concatenate([-a, batched_zeros])
                J[2*c_idx+1] = T.concatenate([ a, batched_zeros])

                if follows_Newtons_third_law:
                    position = positions[:,idx2,:] - positions[:,idx1,:] - parameters['pos_init']
                else:
                    position = positions[:,idx2,:] - parameters['pos_init']

                motor_signal = motor_signals[:,parameters["motor_id"]]
                if "min" in parameters and "max" in parameters:
                    motor_min = (parameters["min"]/180. * np.pi)
                    motor_max = (parameters["max"]/180. * np.pi)
                    motor_signal = T.clip(motor_signal, motor_min, motor_max)

                error_signal = np.dot(position - motor_signal, a)

                if parameters["servo"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["servo"] == "position":
                    if "delta" in parameters and "motor_velocity" in parameters:
                        velocity = parameters["motor_velocity"]
                        b_error[c_idx] = dt * np.clip((abs(error_signal) > parameters["delta"]) * error_signal * parameters["motor_gain"], -velocity, velocity)
                    else:
                        b_error[c_idx] = dt * error_signal * parameters["motor_gain"]

                c_idx += 1


            if constraint == "angular limit":
                # TODO: convert to Theano
                angle = parameters["angle"]/180. * np.pi

                ac = parameters['axis_in_model2_coordinates']
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], rotations[:,idx2,:,:])

                rot_current = theano_dot_last_dimension_matrices(rotations[:,idx2,:,:], rotations[:,idx1,:,:].dimshuffle(0,2,1))
                rot_init = numpy_repeat_new_axis(parameters['rot_init'].T, self.batch_size)
                rot_diff = theano_dot_last_dimension_matrices(rot_current, rot_init)
                
                theta2 = T.arccos(T.clip(0.5*(traces-1),-1+eps,1-eps))
                cross = rot_diff.dimshuffle(0,2,1) - rot_diff
                dot2 = cross[:,1,2] * ac[:,0] + cross[:,2,0] * ac[:,1] + cross[:,0,1] * ac[:,2]

                theta = ((dot2>0) * 2 - 1) * theta2

                if parameters["angle"] < 0:
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),-a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,), dtype=DTYPE), a])
                else:
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,), dtype=DTYPE), a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),-a])

                b_error[c_idx] = np.abs(angle - theta)

                if parameters["angle"] > 0:
                    b_error[c_idx] = angle - theta
                    self.C[c_idx] = (theta > angle)
                else:
                    b_error[c_idx] = theta - angle
                    self.C[c_idx] = (theta < angle)

                c_idx += 1


            if constraint == "linear limit":
                # TODO: convert to Theano
                offset = parameters["offset"]
                if follows_Newtons_third_law:
                    ac = parameters['axis_in_model1_coordinates']
                    a = theano_convert_model_to_world_coordinate_no_bias(ac, self.rot_matrices[idx1,:,:])
                else:
                    a = parameters['axis_in_model1_coordinates']

                if offset < 0:
                    if follows_Newtons_third_law:
                        J[c_idx,0,:] = np.concatenate([-a,np.zeros((3,), dtype=DTYPE)])
                    J[c_idx,1,:] = np.concatenate([a,np.zeros((3,), dtype=DTYPE)])
                else:
                    if follows_Newtons_third_law:
                        J[c_idx,0,:] = np.concatenate([a,np.zeros((3,), dtype=DTYPE)])
                    J[c_idx,1,:] = np.concatenate([-a,np.zeros((3,), dtype=DTYPE)])

                if follows_Newtons_third_law:
                    position = self.positionVectors[idx2,:] - self.positionVectors[idx1,:] - parameters['pos_init']
                else:
                    position = self.positionVectors[idx2,:] - parameters['pos_init']

                current_offset = np.dot(position, a)

                if parameters["offset"] > 0:
                    b_error[c_idx] = offset - current_offset
                    self.C[c_idx] = (current_offset > offset)
                else:
                    b_error[c_idx] = offset - current_offset
                    self.C[c_idx] = (current_offset < offset)

                print current_offset, offset
                c_idx += 1

            if constraint == "ground":
                r = self.radii[idx1].astype('float32')
                J[2*c_idx+0] = numpy_repeat_new_axis(np.array([0,0,1,0,0,0], dtype='float32'), self.batch_size)
                J[2*c_idx+1] = numpy_repeat_new_axis(np.array([0,0,0,0,0,0], dtype='float32'), self.batch_size)

                b_error[c_idx] = T.clip(positions[:,idx1,Z] - r + parameters["delta"], np.finfo('float32').min, 0)
                b_res[c_idx] = parameters["alpha"] * newv[:,idx1,Z]
                C[c_idx] = positions[:,idx1,Z] - r
                c_idx += 1

            if constraint == "ground" and parameters["mu"]!=0:
                r = self.radii[idx1].astype('float32')
                for i in xrange(2):
                    if i==0:
                        J[2*(c_idx+i)+0] = numpy_repeat_new_axis(np.array([0,1,0,-r,0,0], dtype='float32'), self.batch_size)
                    else:
                        J[2*(c_idx+i)+0] = numpy_repeat_new_axis(np.array([1,0,0,0,r,0], dtype='float32'), self.batch_size)
                    J[2*(c_idx+i)+1] = numpy_repeat_new_axis(np.array([0,0,0,0,0,0], dtype='float32'), self.batch_size)
                    C[c_idx+i] = positions[:,idx1,Z] - r
                c_idx += 2

            if constraint == "ground" and parameters["torsional_friction"] and parameters["mu"]!=0:
                r = self.radii[idx1].astype('float32')
                J[2*c_idx+0] = numpy_repeat_new_axis(np.array([0,0,0,0,0,r], dtype='float32'), self.batch_size)
                J[2*c_idx+1] = numpy_repeat_new_axis(np.array([0,0,0,0,0,0], dtype='float32'), self.batch_size)
                C[c_idx] = positions[:,idx1,Z] - r
                c_idx += 1


        zipped_indices = [j for i in zip(self.zero_index,self.one_index) for j in i]

        mass_matrix = M[:,zipped_indices,:,:].reshape(shape=(self.batch_size, self.num_constraints, 2, 6, 6))

        J = theano_stack_batched_integers_mixed_numpy(J, expected_shape=(self.num_constraints,2,self.batch_size,6)).dimshuffle(2,0,1,3)

        C = theano_stack_batched_integers_mixed_numpy(C, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)
        b_res = theano_stack_batched_integers_mixed_numpy(b_res, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)
        b_error = theano_stack_batched_integers_mixed_numpy(b_error, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)

        self.impulses_P = self.warm_start * self.impulses_P

        for iteration in xrange(self.projected_gauss_seidel_iterations):
            # this changes every iteration
            v = newv[:,zipped_indices,:].reshape(shape=(self.batch_size, self.num_constraints, 2, 6))

            # TODO: batch-dot-product
            m_eff = 1./T.sum(T.sum(J[:,:,:,None,:]*mass_matrix, axis=4)*J, axis=(2,3))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)
            ERP = dt*k/(c+dt*k)
            m_c = 1./(1./m_eff + CFM)
            b = ERP/dt * b_error + b_res
            lamb = - m_c * (T.sum(J*v, axis=(2,3)) + CFM * self.impulses_P + b)

            self.impulses_P += lamb
            clipping_force = self.impulses_P[:,self.clipping_idx]

            clipping_limit = abs(self.clipping_a * clipping_force + self.clipping_b * dt)
            self.impulses_P = T.clip(self.impulses_P,-clipping_limit, clipping_limit)
            applicable = (1.0*(C<=0)) * (1-(self.only_when_positive*(self.P<=0)))

            # TODO: batch-dot-product
            result = T.sum(mass_matrix*J[:,:,:,None,:], axis=4) * (self.P * applicable)[:,:,None,None]

            result = result.reshape((self.batch_size, 2*self.num_constraints, 6))

            r = []
            for i in xrange(len(self.map_object_to_constraint)):
                delta_v = T.sum(result[:,self.map_object_to_constraint[i],:], axis=1)
                r.append(delta_v)
            newv = newv + T.stack(r, axis=1)
        #print
        return newv

