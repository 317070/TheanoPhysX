from math import pi

__author__ = 'jonas'
import numpy as np
import theano
import theano.tensor as T

np.seterr(all='raise')

X = 0
Y = 1
Z = 2

def quat_to_rot_matrix(quat):
    w,x,y,z = quat
    wx = w * x * 2.0
    wy = w * y * 2.0
    wz = w * z * 2.0
    xx = x * x * 2.0
    xy = x * y * 2.0
    xz = x * z * 2.0
    yy = y * y * 2.0
    yz = y * z * 2.0
    zz = z * z * 2.0

    return np.array([[ 1 - (yy + zz),        xy - wz,         xz + wy  ],
                    [      xy + wz,    1 - (xx + zz),        yz - wx  ],
                    [      xz - wy,         yz + wx,    1 - (xx + yy) ]], dtype='float32')


def convert_world_to_model_coordinate(coor, model_position, model_orientation):
    return convert_world_to_model_coordinate_no_bias(coor - model_position, model_orientation)

def convert_world_to_model_coordinate_no_bias(coor, model_orientation):
    return np.dot(model_orientation, coor)

def convert_model_to_world_coordinate(coor, rot_matrix, model_position):
    return convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position

def convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    return np.sum(rot_matrix[:,:] * coor[:,None], axis=0)

def theano_convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    return T.sum(rot_matrix * coor[:,None], axis=0)

def numpy_skew_symmetric(x):
    a,b,c = x[...,0,None,None],x[...,1,None,None],x[...,2,None,None]
    z = np.zeros(x.shape[:-1]+(1,1))
    return np.concatenate([
                    np.concatenate(( z,-c, b),axis=-1),
                    np.concatenate(( c, z,-a),axis=-1),
                    np.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)

def single_skew_symmetric(x):
    a,b,c = x[0,None,None],x[1,None,None],x[2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)

def skew_symmetric(x):
    a,b,c = x[:,0,None,None],x[:,1,None,None],x[:,2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)


class TheanoRigid3DBodyEngine(object):
    def __init__(self):
        self.radii = np.zeros(shape=(0,), dtype='float32')
        self.positionVectors = np.zeros(shape=(0,3), dtype='float32')
        self.rot_matrices = np.zeros(shape=(0,3,3), dtype='float32')
        self.velocityVectors = np.zeros(shape=(0,6), dtype='float32')
        self.massMatrices = np.zeros(shape=(0,6,6), dtype='float32')
        self.objects = dict()
        self.constraints = []
        self.num_iterations = 10
        self.sensors = []

        self.P = None
        self.w = None
        self.zeta = None
        self.only_when_positive = None
        self.map_object_to_constraint = None
        self.clipping_a   = None
        self.clipping_idx = None
        self.clipping_b   = None
        self.zero_index = None
        self.one_index = None
        self.inertia_inv = None


    def addCube(self, reference, dimensions, position, rotation, velocity):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, -1)
        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype='float32'), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype='float32'), axis=0)
        mass = 1*np.prod(dimensions)
        I1 = 1./12. * (dimensions[1]**2 + dimensions[2]**2)
        I2 = 1./12. * (dimensions[0]**2 + dimensions[2]**2)
        I3 = 1./12. * (dimensions[0]**2 + dimensions[1]**2)
        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,I1,I2,I3])[None,:,:], axis=0)


    def addSphere(self, reference, radius, position, rotation, velocity):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, radius)
        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype='float32'), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype='float32'), axis=0)
        mass = 1*radius**3

        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,0.4,0.4,0.4])[None,:,:], axis=0)

    def addConstraint(self, constraint, references, parameters):
        references = [self.objects[reference] for reference in references]
        parameters["w"] = parameters["f"] * 2*np.pi

        self.constraints.append([constraint, references, parameters])

    def addGroundConstraint(self, jointname, object1, **parameters):
        self.addConstraint("ground", [object1, object1], parameters)

    def addTouchConstraint(self, jointname, object1, object2, **parameters):
        self.addConstraint("touch", [object1, object2], parameters)


    def addBallAndSocketConstraint(self, jointname, object1, object2, point, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:], self.rot_matrices[idx1,:,:]).astype('float32')
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:], self.rot_matrices[idx2,:,:]).astype('float32')

        self.addConstraint("ball-and-socket", [object1, object2], parameters)


    def addHingeConstraint(self, jointname, object1, object2, point, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:], self.rot_matrices[idx1,:,:]).astype('float32')
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:], self.rot_matrices[idx2,:,:]).astype('float32')

        # create two forbidden axis:
        axis = np.array(axis, dtype='float32')
        axis = axis / np.linalg.norm(axis)
        if (axis == np.array([1,0,0])).all():
            forbidden_axis_1 = np.array([0,1,0], dtype='float32')
            forbidden_axis_2 = np.array([0,0,1], dtype='float32')
        else:
            forbidden_axis_1 = np.array([0,-axis[2],axis[1]], dtype='float32')
            forbidden_axis_2 = np.cross(axis, forbidden_axis_1)

        parameters['axis'] = axis.astype('float32')
        parameters['axis1_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_1, self.rot_matrices[idx1,:]).astype('float32')
        parameters['axis2_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_2, self.rot_matrices[idx1,:]).astype('float32')
        parameters['axis_in_model2_coordinates'] =  convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx2,:]).astype('float32')

        self.addConstraint("hinge", [object1, object2], parameters)



    def addSliderConstraint(self, jointname, object1, object2, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        self.addConstraint("slider", [object1, object2], parameters)

    def addFixedConstraint(self, jointname, object1, object2, point, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:], self.rot_matrices[idx1,:,:]).astype('float32')
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:], self.rot_matrices[idx2,:,:]).astype('float32')

        self.addConstraint("fixed", [object1, object2], parameters)

    def addMotorConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        # create two forbidden axis:
        axis = np.array(axis, dtype='float32')
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis.astype('float32')
        parameters['axis_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx1,:,:]).astype('float32')

        self.addConstraint("motor", [object1, object2], parameters)


    def addLimitConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        axis = np.array(axis, dtype='float32')
        axis = axis / np.linalg.norm(axis)

        axis = np.array(axis, dtype='float32')
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis.astype('float32')
        parameters['axis_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx1,:,:]).astype('float32')

        self.addConstraint("limit", [object1, object2], parameters)


    def addSensor(self, object, reference_object):
        self.sensors.append((object,reference_object))



    def compile(self):

        self.num_constraints = 0
        for (constraint,references,parameters) in self.constraints:
            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                self.num_constraints += 3
            if constraint == "slider" or constraint == "fixed":
                self.num_constraints += 3
            if constraint == "hinge":
                self.num_constraints += 2
            if constraint == "hinge" and "limit" in parameters:
                self.num_constraints += 2
            if constraint == "hinge" and "velocity_motor" in parameters:
                self.num_constraints += 1
            if constraint == "hinge" and "position_motor" in parameters:
                self.num_constraints += 1
            if constraint == "ground":
                self.num_constraints += 1
            if constraint == "ground" and parameters["mu"]!=0:
                self.num_constraints += 2
            if constraint == "ground" and parameters["torsional_friction"]:
                self.num_constraints += 1
            if constraint == "limit":
                self.num_constraints += 1
            if constraint == "motor":
                self.num_constraints += 1

        self.P = T.zeros((self.num_constraints,))
        self.w = np.zeros((self.num_constraints,), dtype='float32')  # 0 constraints
        self.zeta = np.zeros((self.num_constraints,), dtype='float32')  # 0 constraints

        self.only_when_positive = np.zeros((self.num_constraints,), dtype='float32')  # 0 constraints
        self.map_object_to_constraint = [[] for _ in xrange(len(self.objects))]
        self.clipping_a   = np.ones((self.num_constraints,), dtype='float32')
        self.clipping_idx = range(self.num_constraints)
        self.clipping_b   = np.zeros((self.num_constraints,), dtype='float32')

        self.zero_index = []
        self.one_index = []
        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            idx2 = references[1]

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                for i in xrange(3):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)
                    self.zero_index.append(idx1)
                    self.one_index.append(idx2)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 3

            if constraint == "slider" or constraint == "fixed":
                parameters['rot_init'] = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)

                for i in xrange(3):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)
                    self.zero_index.append(idx1)
                    self.one_index.append(idx2)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 3


            if constraint == "hinge":
                for i in xrange(2):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)
                    self.zero_index.append(idx1)
                    self.one_index.append(idx2)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 2

            if constraint == "limit":
                parameters['rot_init'] = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)
                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.map_object_to_constraint[idx2].append(2*c_idx + 1)
                self.zero_index.append(idx1)
                self.one_index.append(idx2)

                self.only_when_positive[c_idx] = 1.0
                self.w[c_idx] = parameters["f"] * 2*np.pi
                self.zeta[c_idx] = parameters["zeta"]
                c_idx += 1

            if constraint == "motor":
                parameters['rot_init'] = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)

                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.map_object_to_constraint[idx2].append(2*c_idx + 1)
                self.zero_index.append(idx1)
                self.one_index.append(idx2)

                if "motor_torque" in parameters:
                    self.clipping_a[c_idx] = 0
                    self.clipping_b[c_idx] = parameters["motor_torque"]
                self.w[c_idx] = parameters["f"] * 2*np.pi
                self.zeta[c_idx] = parameters["zeta"]
                c_idx += 1

            if constraint == "ground":
                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.w[c_idx] = parameters["f"] * 2*np.pi
                self.zeta[c_idx] = parameters["zeta"]
                self.only_when_positive[c_idx] = 1.0
                ground_contact_idx = c_idx
                c_idx += 1
                self.zero_index.append(idx1)
                self.one_index.append(idx2)

            if constraint == "ground" and parameters["mu"]!=0:
                for i in xrange(2):
                    self.clipping_a[c_idx+i] = parameters["mu"]
                    self.clipping_idx[c_idx+i] = ground_contact_idx
                    self.clipping_b[c_idx+i] = 0
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.w[c_idx+i] = parameters["f"] * 2*np.pi
                    self.zeta[c_idx+i] = parameters["zeta"]
                    self.zero_index.append(idx1)
                    self.one_index.append(idx2)

                c_idx += 2

            if constraint == "ground" and parameters["torsional_friction"]:
                d = parameters["delta"]
                r = self.radii[idx1]
                self.clipping_a[c_idx] = 3.*np.pi/16. * np.sqrt(r*d) * parameters["mu"]
                self.clipping_idx[c_idx] = ground_contact_idx
                self.clipping_b[c_idx] = 0
                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.w[c_idx] = parameters["f"] * 2*np.pi
                self.zeta[c_idx] = parameters["zeta"]
                self.zero_index.append(idx1)
                self.one_index.append(idx2)

                c_idx += 1

        self.positionVectors = theano.shared(self.positionVectors.astype('float32'), name="positionVectors")
        self.velocityVectors = theano.shared(self.velocityVectors.astype('float32'), name="velocityVectors")
        self.rot_matrices = theano.shared(self.rot_matrices.astype('float32'), name="rot_matrices", )
        self.inertia_inv = theano.shared(np.linalg.inv(self.massMatrices).astype('float32'), name="inertia_inv", )




    def evaluate(self, dt, positions, velocities, rot_matrices, motor_signals):

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!
        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,0,0,0,0], dtype='float32')  # total force acting on body outside of constraints
        acceleration = np.array([0,0,-9.81,0,0,0], dtype='float32')  # acceleration of the default frame
        newv = velocities + dt * acceleration[None,:]
        originalv = newv


        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        # convert mass matrices to world coordinates
        M = T.zeros_like(self.inertia_inv)


        M00 = self.inertia_inv[:,:3,:3]
        M01 = T.zeros_like(M00)
        M10 = M01
        M11 = T.sum(rot_matrices.dimshuffle(0,1,'x',2,'x') * self.inertia_inv[:,3:,3:].dimshuffle(0,1,2,'x','x') * rot_matrices.dimshuffle(0,'x',1,'x',2), axis=(1,2))
        M0 = T.concatenate([M00,M01],axis=-1)
        M1 = T.concatenate([M10,M11],axis=-1)
        M = T.concatenate([M0,M1],axis=-2)

        #self.P = np.zeros((self.num_constraints,))# constant
        # changes every timestep
        J = [np.zeros((1,6)) for _ in xrange(2 * self.num_constraints)]  # 0 constraints x 2 objects x 6 states
        b_res = [0 for _ in xrange(self.num_constraints)]  # 0 constraints
        b_error = [0 for _ in xrange(self.num_constraints)]  # 0 constraints
        C = [1 for _ in xrange(self.num_constraints)]  # 0 constraints
        mass_matrix = [np.zeros((2,6,6)) for _ in xrange(self.num_constraints)]  # 0 constraints x 2 objects x 6 states x 6 states

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            if references[1] is not None:
                idx2 = references[1]
            else:
                idx2 = None

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                r1x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"], rot_matrices[idx1,:,:])
                r2x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"], rot_matrices[idx2,:,:])
                ss_r1x = single_skew_symmetric(r1x)
                ss_r2x = single_skew_symmetric(r2x)
                complete_J1 = T.concatenate([-np.eye(3, dtype='float32'), ss_r1x])
                complete_J2 = T.concatenate([ np.eye(3, dtype='float32'),-ss_r2x])
                error = positions[idx2,:]+r2x-positions[idx1,:]-r1x
                for i in xrange(3):
                    J[2*(c_idx+i)+0] = complete_J1[:,i]
                    J[2*(c_idx+i)+1] = complete_J2[:,i]

                    b_error[c_idx+i] = error[i]
                c_idx += 3

            if constraint == "slider" or constraint == "fixed":
                complete_J1 = np.concatenate([np.zeros((3,3)),-np.eye(3)]).astype('float32')
                complete_J2 = np.concatenate([np.zeros((3,3)), np.eye(3)]).astype('float32')

                for i in xrange(3):
                    J[2*(c_idx+i)+0] = complete_J1[:,i]
                    J[2*(c_idx+i)+1] = complete_J2[:,i]

                rot_current = np.dot(rot_matrices[idx2,:,:], rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, parameters['rot_init'].T)
                cross = rot_diff.T - rot_diff

                b_error[c_idx] = 0#cross[1,2]
                b_error[c_idx+1] = 0#cross[2,0]
                b_error[c_idx+2] = 0#cross[0,1]
                c_idx += 3

            if constraint == "hinge":
                a2x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], rot_matrices[idx2,:,:])
                b1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], rot_matrices[idx1,:,:])
                c1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], rot_matrices[idx1,:,:])
                ss_a2x = single_skew_symmetric(a2x)

                J[2*(c_idx+0)+0] = T.concatenate([np.zeros((3,), dtype='float32'),-T.dot(b1x,ss_a2x)])
                J[2*(c_idx+0)+1] = T.concatenate([np.zeros((3,), dtype='float32'), T.dot(b1x,ss_a2x)])
                J[2*(c_idx+1)+0] = T.concatenate([np.zeros((3,), dtype='float32'),-T.dot(c1x,ss_a2x)])
                J[2*(c_idx+1)+1] = T.concatenate([np.zeros((3,), dtype='float32'), T.dot(c1x,ss_a2x)])

                b_error[c_idx+0] = T.sum(a2x*b1x)
                b_error[c_idx+1] = T.sum(a2x*c1x)
                c_idx += 2

            if constraint == "limit":
                angle = parameters["angle"]/180. * np.pi
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], rot_matrices[idx1,:,:])
                rot_current = np.dot(rot_matrices[idx2,:,:], rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, quat_to_rot_matrix(parameters['rot_init']).T)
                theta2 = np.arccos(0.5*(np.trace(rot_diff)-1))
                cross = rot_diff.T - rot_diff
                dot2 = cross[1,2] * a[0] + cross[2,0] * a[1] + cross[0,1] * a[2]
                theta = ((dot2>0) * 2 - 1) * theta2

                if parameters["angle"] < 0:
                    J[2*c_idx+0] = np.concatenate([np.zeros((3,), dtype='float32'),-a])
                    J[2*c_idx+1] = np.concatenate([np.zeros((3,), dtype='float32'), a])
                else:
                    J[2*c_idx+0] = np.concatenate([np.zeros((3,), dtype='float32'), a])
                    J[2*c_idx+1] = np.concatenate([np.zeros((3,), dtype='float32'),-a])

                b_error[c_idx] = np.abs(angle - theta)
                if parameters["angle"] > 0:
                    b_error[c_idx] = angle - theta
                    C[c_idx] = (theta > angle)
                else:
                    b_error[c_idx] = theta - angle
                    C[c_idx] = (theta < angle)

                c_idx += 1

            if constraint == "motor":
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], rot_matrices[idx1,:,:])

                rot_current = T.dot(rot_matrices[idx2,:,:], rot_matrices[idx1,:,:].T)
                rot_diff = T.dot(rot_current, parameters['rot_init'].T)
                theta2 = T.arccos(T.clip(0.5*(T.nlinalg.trace(rot_diff)-1),-1,1))
                cross = rot_diff.T - rot_diff
                dot2 = cross[1,2] * a[0] + cross[2,0] * a[1] + cross[0,1] * a[2]

                theta = ((dot2>0) * 2 - 1) * theta2

                J[2*c_idx+0] = T.concatenate([np.zeros((3,), dtype='float32'),-a])
                J[2*c_idx+1] = T.concatenate([np.zeros((3,), dtype='float32'), a])

                motor_signal = motor_signals[parameters["motor_id"]]
                motor_min = (parameters["min"]/180. * np.pi)
                motor_max = (parameters["max"]/180. * np.pi)
                motor_signal = np.clip(motor_signal, motor_min, motor_max).astype('float32')

                if parameters["type"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["type"] == "position":
                    if "delta" in parameters:
                        b_error[c_idx] = dt * (abs(theta-motor_signal) > parameters["delta"]) * (2*(theta>motor_signal)-1) * parameters["motor_velocity"]
                    else:
                        b_error[c_idx] = dt * (theta-motor_signal) * parameters["motor_velocity"]

                #print c_idx

                c_idx += 1

            if constraint == "ground":
                r = self.radii[idx1].astype('float32')
                J[2*c_idx+0] = np.array([0,0,1,0,0,0], dtype='float32')
                J[2*c_idx+1] = np.array([0,0,0,0,0,0], dtype='float32')

                b_error[c_idx] = T.clip(positions[idx1,Z] - r + parameters["delta"], np.finfo('float32').min, 0)
                b_res[c_idx] = parameters["alpha"] * newv[idx1,Z]
                C[c_idx] = (positions[idx1,Z] - r < 0.0)
                c_idx += 1

            if constraint == "ground" and parameters["mu"]!=0:
                r = self.radii[idx1].astype('float32')
                for i in xrange(2):
                    if i==0:
                        J[2*(c_idx+i)+0] = np.array([0,1,0,-r,0,0], dtype='float32')
                    else:
                        J[2*(c_idx+i)+0] = np.array([1,0,0,0,r,0], dtype='float32')
                    J[2*(c_idx+i)+1] = np.array([0,0,0,0,0,0], dtype='float32')
                    C[c_idx+i] = (positions[idx1,Z] - r < 0.0)
                c_idx += 2

            if constraint == "ground" and parameters["torsional_friction"]:
                r = self.radii[idx1].astype('float32')
                J[2*c_idx+0] = np.array([0,0,0,0,0,r], dtype='float32')
                J[2*c_idx+1] = np.array([0,0,0,0,0,0], dtype='float32')
                C[c_idx] = (positions[idx1,Z] - r < 0.0)
                b_res[c_idx] = 0
                c_idx += 1

        mass_matrix = T.concatenate((M[self.zero_index,None,:,:], M[self.one_index,None,:,:]), axis=1)
        J = T.stack(J, axis=0).reshape(shape=(self.num_constraints,2,6))
        C = T.stack(C, axis=0)
        #v = np.zeros((self.num_constraints,2,6))  # 0 constraints x 2 objects x 6 states

        for iteration in xrange(self.num_iterations):
            # changes every iteration
            v = T.concatenate((newv[self.zero_index,None,:],newv[self.one_index,None,:]),axis=1)

            m_eff = 1./T.sum(T.sum(J[:,:,None,:]*mass_matrix, axis=-1)*J, axis=(-1,-2))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)
            ERP = dt*k/(c+dt*k)

            m_c = 1./(1./m_eff + CFM)

            b = ERP/dt * b_error + b_res
            lamb = - m_c * (T.sum(J*v, axis=(-1,-2)) + CFM * self.P + b)

            self.P += lamb
            #print J[[39,61,65,69],:]
            #print np.sum(lamb**2), np.sum(self.P**2)
            clipping_limit = abs(self.clipping_a * self.P[self.clipping_idx] + self.clipping_b * dt)
            self.P = T.clip(self.P,-clipping_limit, clipping_limit)
            applicable = (1-(self.only_when_positive*( 1-(self.P>=0)) )) * C

            result = T.sum(mass_matrix*J[:,:,None,:], axis=(-1)) * self.P[:,None,None] * applicable[:,None,None]
            result = result.reshape((2*self.num_constraints,6))

            r = []
            for i in xrange(len(self.map_object_to_constraint)):
                r.append(originalv[i,:] + T.sum(result[self.map_object_to_constraint[i],:], axis=0))
            newv = T.stack(r, axis=0)

        #print
        return newv


    def do_time_step(self, dt=1e-3, motor_signals=list()):
        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.

        # semi-implicit Euler integration
        self.velocityVectors = self.evaluate(dt, self.positionVectors, self.velocityVectors, self.rot_matrices, motor_signals=motor_signals)

        self.positionVectors = self.positionVectors[:,:] + self.velocityVectors[:,:3] * dt
        self.rot_matrices = normalize_matrix(self.rot_matrices[:,:,:] + T.sum(self.rot_matrices[:,:,:,None] * skew_symmetric(dt * self.velocityVectors[:,3:])[:,None,:,:],axis=2) )

    def step_from_this_state(self, state, dt=1e-3, motor_signals=list()):
        positions, velocities, rot_matrices = state
        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.

        # semi-implicit Euler integration
        velocities = self.evaluate(dt, positions, velocities, rot_matrices, motor_signals=motor_signals)

        positions = positions + velocities[:,:3] * dt
        rot_matrices = normalize_matrix(rot_matrices[:,:,:] + T.sum(rot_matrices[:,:,:,None] * skew_symmetric(dt * velocities[:,3:])[:,None,:,:],axis=2) )

        return (positions, velocities, rot_matrices)



    def getState(self):
        return self.positionVectors, self.velocityVectors, self.rot_matrices

    def getPosition(self, reference):
        idx = self.getObjectIndex(reference)
        return self.positionVectors[idx]

    def getRotationMatrix(self, reference):
        idx = self.getObjectIndex(reference)
        return self.rot_matrices[idx]

    def getObjectIndex(self, reference):
        return self.objects[reference]

    def getSensorValues(self, reference):
        # make positionvectors neutral according to reference object
        idx = self.objects[reference]
        rot_matrix = self.rot_matrices[idx]

        ref_position = self.positionVectors[idx,:]

        result = self.positionVectors
        #result[:,:2] = result[:,:2]-ref_position[None,:2]
        # step 1: flatten rot_matrix


        # step 2: apply rot_matrix on positions and rotations



        #result = result.flatten() + self.C[self.only_when_positive==1]

        return result



def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    res = np.array([w, x, y, z], dtype='float32').T
    return res

def q_inv(q):
    w, x, y, z = q
    return [w,-x,-y,-z]


def q_div(q1, q2):
    w, x, y, z = q2
    return q_mult([w,-x,-y,-z], q1)

def normalize(q):
    return q/np.linalg.norm(q, axis=-1, keepdims=True)


def normalize_matrix(A):
    for i in xrange(1):
        A = (3*A - T.sum(A[:,:,None,:,None] * A[:,None,:,:,None] * A[:,None,:,None,:], axis=(-3,-2))) / 2
    return A
