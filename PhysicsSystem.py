from math import pi

__author__ = 'jonas'
import numpy as np
import scipy.linalg
import random

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
                    [      xz - wy,         yz + wx,    1 - (xx + yy) ]])


def convert_world_to_model_coordinate(coor, model_position):
    return np.dot(quat_to_rot_matrix(model_position[3:]), coor - model_position[:3])

def convert_world_to_model_coordinate_no_bias(coor, model_position):
    return np.dot(quat_to_rot_matrix(model_position[3:]), coor)

def convert_model_to_world_coordinate(coor, model_position):
    return np.dot(quat_to_rot_matrix(model_position[3:]).T, coor) + model_position[:3]

def convert_model_to_world_coordinate_no_bias(coor, model_position):
    return np.dot(quat_to_rot_matrix(model_position[3:]).T, coor)

def skew_symmetric(x):
    a,b,c = x
    return np.array([[ 0,-c, b],
                    [  c, 0,-a ],
                    [ -b, a, 0 ]])


def vec_vec_dot(a,b):
    return np.tensordot(a,b, axes=([-1],[-1]))

def vec_mat_dot(a,b):
    return np.tensordot(a,b, axes=([-1],[-2]))

def mat_vec_dot(a,b):
    return np.tensordot(a,b, axes=([-1],[-1]))


class Rigid3DBodyEngine(object):
    def __init__(self):
        self.positionVectors = np.zeros(shape=(0,7))
        self.velocityVectors = np.zeros(shape=(0,6))
        self.massMatrices = np.zeros(shape=(0,6,6))
        self.objects = dict()
        self.constraints = []
        self.num_iterations = 1

    def addSphere(self, reference, position, velocity):
        self.objects[reference] = self.positionVectors.shape[0]

        self.positionVectors = np.append(self.positionVectors, np.array([position]), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity]), axis=0)
        self.massMatrices = np.append(self.massMatrices, np.diag([1,1,1,0.4,0.4,0.4])[None,:,:], axis=0)

    def addConstraint(self, constraint, references, parameters):
        references = [self.objects[reference] for reference in references]
        parameters["CFM"] = 1.0
        parameters["ERP"] = 0.8
        self.constraints.append([constraint, references, parameters])

    def addGroundConstraint(self, object, parameters):
        self.addConstraint("ground", [object, object], parameters)

    def addBallAndSocketConstraint(self, object1, object2, world_coordinates, parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx2,:])

        self.addConstraint("ball-and-socket", [object1, object2], parameters)


    def addHingeConstraint(self, object1, object2, world_coordinates, axis, parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx2,:])

        # create two forbidden axis:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        if (axis == np.array([1,0,0])).all():
            forbidden_axis_1 = np.array([0,1,0])
            forbidden_axis_2 = np.array([0,0,1])
        else:
            forbidden_axis_1 = np.array([0,-axis[2],axis[1]])
            forbidden_axis_2 = np.cross(axis, forbidden_axis_1)

        parameters['axis'] = axis
        parameters['axis1_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_1, self.positionVectors[idx1,:])
        parameters['axis2_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_2, self.positionVectors[idx1,:])
        parameters['axis_in_model2_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.positionVectors[idx2,:])

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])


        self.addConstraint("hinge", [object1, object2], parameters)



    def addSliderConstraint(self, object1, object2, parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("slider", [object1, object2], parameters)

    def addFixedConstraint(self, object1, object2, world_coordinates, parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(world_coordinates, self.positionVectors[idx2,:])

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("fixed", [object1, object2], parameters)



    def do_time_step(self, dt=1e-3):

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!

        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,-9.81,0,0,0])  # total force acting on body outside of constraints
        newv = self.velocityVectors + dt * np.dot(self.massMatrices, totalforce)
        originalv = newv.copy()


        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        M = np.linalg.inv(self.massMatrices)

        num_constraints = 0
        for (constraint,references,parameters) in self.constraints:
            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                num_constraints += 3
            if constraint == "slider" or constraint == "fixed":
                num_constraints += 3
            if constraint == "hinge":
                num_constraints += 2
            if constraint == "hinge" and "limit" in parameters:
                num_constraints += 2
            if constraint == "hinge" and "velocity_motor" in parameters:
                num_constraints += 1
            if constraint == "hinge" and "position_motor" in parameters:
                num_constraints += 1
            if constraint == "ground":
                num_constraints += 1
            if constraint == "ground" and parameters["mu"]!=0:
                num_constraints += 2
            if constraint == "ground" and parameters["torsional_friction"]:
                num_constraints += 1

        print num_constraints


        # TODO: (J * invM * JT + CFM) * dP = -(J * v2damaged + CFM * P + bias)
        # http://bulletphysics.org/Bullet/phpBB3/viewtopic.php?f=4&t=1354
        # ERP and CFM

        # TODO: drop quaternions, use rotation matrices!

        # TODO: fix contact joints for arbitrary radiuses
        # TODO: check J-matrix in friction, I guess there is something wrong

        P = np.zeros((num_constraints,))

        for iteration in xrange(self.num_iterations):

            total_lambda = np.zeros_like(self.velocityVectors)

            # changes every timestep
            J = np.zeros((num_constraints,2,6))  # 0 constraints x 2 objects x 6 states
            v = np.zeros((num_constraints,2,6))  # 0 constraints x 2 objects x 6 states
            b_res = np.zeros((num_constraints,))  # 0 constraints
            b_error = np.zeros((num_constraints,))  # 0 constraints

            # constant
            mass_matrix = np.zeros((num_constraints,2,6,6))  # 0 constraints x 2 objects x 6 states x 6 states
            CFM = np.zeros((num_constraints,))  # 0 constraints
            ERP = np.zeros((num_constraints,))  # 0 constraints
            only_when_positive = np.zeros((num_constraints,))  # 0 constraints
            C = np.ones((num_constraints,))  # 0 constraints
            map_object_to_constraint = [[] for _ in xrange(newv.shape[0])]
            clipping_a   = np.ones((num_constraints,))
            clipping_idx = range(num_constraints)
            clipping_b   = np.zeros((num_constraints,))

            c_idx = 0
            for constraint,references,parameters in self.constraints:
                idx1 = references[0]
                idx2 = references[1]

                if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":

                    r1x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"], self.positionVectors[idx1,:])
                    r2x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"], self.positionVectors[idx2,:])
                    for i in xrange(3):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                        J[c_idx+i,0,:] = np.concatenate([-np.eye(3), skew_symmetric(r1x)])[:,i]
                        J[c_idx+i,1,:] = np.concatenate([ np.eye(3),-skew_symmetric(r2x)])[:,i]
                        mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                        mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]
                        map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                        map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                    b_error[c_idx:c_idx+3] = (self.positionVectors[idx2,:3]+r2x-self.positionVectors[idx1,:3]-r1x)
                    CFM[c_idx:c_idx+3] = parameters["CFM"]
                    ERP[c_idx:c_idx+3] = parameters["ERP"]
                    c_idx += 3

                if constraint == "slider" or constraint == "fixed":

                    for i in xrange(3):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                        J[c_idx+i,0,:] = np.concatenate([np.zeros((3,3)),-np.eye(3)])[:,i]
                        J[c_idx+i,1,:] = np.concatenate([np.zeros((3,3)), np.eye(3)])[:,i]
                        mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                        mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]
                        map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                        map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                    q_current = normalize(q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:]))
                    q_diff = q_div(q_current, parameters['q_init'])
                    b_error[c_idx:c_idx+3] = 2*q_diff[1:]
                    CFM[c_idx:c_idx+3] = parameters["CFM"]
                    ERP[c_idx:c_idx+3] = parameters["ERP"]
                    c_idx += 3


                if constraint == "hinge":
                    a2x = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], self.positionVectors[idx2,:])
                    b1x = convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], self.positionVectors[idx1,:])
                    c1x = convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], self.positionVectors[idx1,:])
                    for i in xrange(2):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                        if i==0:
                            J[c_idx+i,0,:] = np.concatenate([np.zeros((3,)),-np.dot(skew_symmetric(a2x),b1x)])
                            J[c_idx+i,1,:] = np.concatenate([np.zeros((3,)), np.dot(skew_symmetric(a2x),b1x)])
                        else:
                            J[c_idx+i,0,:] = np.concatenate([np.zeros((3,)),-np.dot(skew_symmetric(a2x),c1x)])
                            J[c_idx+i,1,:] = np.concatenate([np.zeros((3,)), np.dot(skew_symmetric(a2x),c1x)])
                        mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                        mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]
                        map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                        map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                    b_error[c_idx:c_idx+2] =np.array([np.sum(a2x*b1x),np.sum(a2x*c1x)])
                    CFM[c_idx:c_idx+3] = parameters["CFM"]
                    ERP[c_idx:c_idx+3] = parameters["ERP"]

                    c_idx += 2

                if constraint == "hinge" and "limit" in parameters:
                    q_current = normalize(q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:]))
                    q_diff = q_div(q_current, parameters['q_init'])
                    a=parameters['axis']
                    dot = np.sum(q_diff[1:] * a)
                    sin_theta2 = ((dot>0) * 2 - 1) * np.sqrt(np.sum(q_diff[1:]*q_diff[1:]))
                    theta = 2*np.arctan2(sin_theta2,q_diff[0])

                    for i in xrange(2):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                        if i==0:
                            J[c_idx+i,0,:] = np.concatenate([np.zeros((3,)),-a])
                            J[c_idx+i,1,:] = np.concatenate([np.zeros((3,)), a])
                        else:
                            J[c_idx+i,0,:] = np.concatenate([np.zeros((3,)), a])
                            J[c_idx+i,1,:] = np.concatenate([np.zeros((3,)),-a])
                        mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                        mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]
                        map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                        map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                    b_error[c_idx:c_idx+2] = np.array([(parameters["limit"] - theta),(theta - parameters["limit"])])
                    C[c_idx:c_idx+2] = np.array([(theta<-parameters["limit"]), (theta>parameters["limit"])])
                    only_when_positive[c_idx:c_idx+2] = 1.0
                    CFM[c_idx:c_idx+2] = parameters["CFM"]
                    ERP[c_idx:c_idx+2] = parameters["ERP"]

                    c_idx += 2

                if constraint == "hinge" and "velocity_motor" in parameters:

                    a=parameters['axis']
                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = newv[idx2,:]
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,)),-a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,)), a])

                    mass_matrix[c_idx,0,:,:] = M[idx1,:,:]
                    mass_matrix[c_idx,1,:,:] = M[idx2,:,:]
                    map_object_to_constraint[idx1].append(2*c_idx + 0)
                    map_object_to_constraint[idx2].append(2*c_idx + 1)

                    b_error[c_idx] = parameters["motor_velocity"]
                    clipping_a[c_idx] = 0
                    clipping_b[c_idx] = parameters["motor_torque"]
                    CFM[c_idx] = parameters["CFM"]
                    ERP[c_idx] = parameters["ERP"]

                    c_idx += 1

                if constraint == "hinge" and "position_motor" in parameters:
                    a=parameters['axis']
                    dot = np.sum(q_diff[1:] * a)
                    sin_theta2 = np.sqrt(np.sum(q_diff[1:]*q_diff[1:]))
                    theta = 2*((dot>0) * 2 - 1)*np.arctan2(sin_theta2,q_diff[0])

                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = newv[idx2,:]
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,)),-a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,)), a])

                    mass_matrix[c_idx,0,:,:] = M[idx1,:,:]
                    mass_matrix[c_idx,1,:,:] = M[idx2,:,:]
                    map_object_to_constraint[idx1].append(2*c_idx + 0)
                    map_object_to_constraint[idx2].append(2*c_idx + 1)

                    b_error[c_idx] = (abs(theta-parameters["motor_position"]) > parameters["delta"]) * (2*(theta>parameters["motor_position"])-1) * parameters["motor_velocity"]
                    clipping_a[c_idx] = 0
                    clipping_b[c_idx] = parameters["motor_torque"]

                    c_idx += 1

                if constraint == "ground":
                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = np.array([0,0,0,0,0,0])
                    J[c_idx,0,:] = np.array([0,0,1,0,0,0])
                    J[c_idx,1,:] = np.array([0,0,0,0,0,0])
                    b_error[c_idx] = np.max(self.positionVectors[idx1,Z] - parameters["delta"],0)
                    b_res[c_idx] = parameters["alpha"] * newv[idx1,Z]
                    C[c_idx] = (self.positionVectors[idx1,Z] < 0.0)
                    only_when_positive[c_idx] = 1.0
                    ground_contact_idx = c_idx
                    c_idx += 1

                if constraint == "ground" and parameters["mu"]!=0:
                    for i in xrange(2):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = np.array([0,0,0,0,0,0])
                        if i==0:
                            J[c_idx+i,0,:] = np.array([0,1,0,1,0,0])
                        else:
                            J[c_idx+i,0,:] = np.array([1,0,0,0,1,0])
                        J[c_idx+i,1,:] = np.array([0,0,0,0,0,0])
                        clipping_a[c_idx+i] = parameters["mu"]
                        clipping_idx[c_idx+i] = ground_contact_idx
                        clipping_b[c_idx+i] = 0
                        C[c_idx+i] = (self.positionVectors[idx1,Z] < 0.0)
                        b_res[c_idx+i] = 0
                    c_idx += 2

                if constraint == "ground" and parameters["torsional_friction"]:
                    d = parameters["delta"]
                    r = 1 #parameters["radius"]
                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = np.array([0,0,0,0,0,0])
                    J[c_idx+i,0,:] = np.array([0,0,0,0,0,1])
                    J[c_idx+i,1,:] = np.array([0,0,0,0,0,0])
                    clipping_a[c_idx+i] = 3.*np.pi/16. * np.sqrt(r*d) * parameters["mu"]
                    clipping_idx[c_idx+i] = ground_contact_idx
                    clipping_b[c_idx+i] = 0
                    C[c_idx+i] = (self.positionVectors[idx1,Z] < 0.0)
                    b_res[c_idx+i] = 0
                    c_idx += 1

            m_eff = 1./np.sum(np.sum(J[:,:,None,:]*M, axis=-1)*J, axis=(-1,-2))
            m_c = 1/(1/m_eff + CFM)

            #print b_res
            b = ERP/dt * b_error + b_res

            lamb = - m_c * (np.sum(J*v, axis=(-1,-2)) + CFM * P + b)
            print lamb
            P += lamb

            #print v.shape, np.dot(mass_matrix, J.T).shape, lamb.shape,
            clipping_limit = np.abs(clipping_a * P[clipping_idx] + clipping_b)

            P = np.clip(P,-clipping_limit, clipping_limit)

            #result = np.dot(np.dot(mass_matrix, J.T), P[c_idx][1])
            # applicable should be zero when (P is negative, but only when only_when_positive is 1), or when C is zero
            applicable = (1-(only_when_positive*( 1-( P<0)) )) * C

            result = np.sum(M*J[:,:,None,:], axis=(-1)) * P[:,None,None] * applicable[:,None,None]

            result = result.reshape(result.shape[:-3] + (2*num_constraints,6))

            for i in xrange(newv.shape[0]):
                newv[i,:] = originalv[i,:] + np.sum(result[map_object_to_constraint[i],:], axis=0)


        print
        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.
        self.velocityVectors = newv

        self.positionVectors[:,:3] = self.positionVectors[:,:3] + self.velocityVectors[:,:3] * dt

        v_norm = np.linalg.norm(self.velocityVectors[:,3:], axis=-1)
        a = self.velocityVectors[:,3:] / (v_norm + 1e-13)[:,None]
        theta = v_norm*dt
        self.positionVectors[:,3:] = normalize(q_mult(self.positionVectors[:,3:].T, [np.cos(theta/2), a[:,0]*np.sin(theta/2), a[:,1]*np.sin(theta/2), a[:,2]*np.sin(theta/2)]))


    def getPosition(self, reference):
        idx = self.objects[reference]
        return self.positionVectors[idx]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    res = np.array([w, x, y, z]).T
    return res

def q_div(q1, q2):
    w, x, y, z = q2
    return q_mult([w,-x,-y,-z], q1)

def normalize(q):
    return q/np.linalg.norm(q, axis=-1, keepdims=True)
