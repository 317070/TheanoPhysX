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

def convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    return np.dot(rot_matrix.T, coor)

def skew_symmetric(x):
    a,b,c = x
    return np.array([[ 0,-c, b],
                    [  c, 0,-a ],
                    [ -b, a, 0 ]])


class Rigid3DBodyEngine(object):
    def __init__(self):
        self.radii = np.zeros(shape=(0,))
        self.positionVectors = np.zeros(shape=(0,7))
        self.velocityVectors = np.zeros(shape=(0,6))
        self.massMatrices = np.zeros(shape=(0,6,6))
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
        self.C = None


        self.rot_matrices = None


    def addCube(self, reference, dimensions, position, velocity):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, -1)
        self.positionVectors = np.append(self.positionVectors, np.array([position]), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity]), axis=0)
        mass = 1*np.prod(dimensions)
        I1 = 1./12. * (dimensions[1]**2 + dimensions[2]**2)
        I2 = 1./12. * (dimensions[0]**2 + dimensions[2]**2)
        I3 = 1./12. * (dimensions[0]**2 + dimensions[1]**2)
        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,I1,I2,I3])[None,:,:], axis=0)


    def addSphere(self, reference, radius, position, velocity):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, radius)
        self.positionVectors = np.append(self.positionVectors, np.array([position]), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity]), axis=0)
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

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:])

        self.addConstraint("ball-and-socket", [object1, object2], parameters)


    def addHingeConstraint(self, jointname, object1, object2, point, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:])

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



    def addSliderConstraint(self, jointname, object1, object2, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("slider", [object1, object2], parameters)

    def addFixedConstraint(self, jointname, object1, object2, point, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.positionVectors[idx2,:])

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("fixed", [object1, object2], parameters)

    def addMotorConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        # create two forbidden axis:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis
        parameters['axis_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.positionVectors[idx1,:])
        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("motor", [object1, object2], parameters)


    def addLimitConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis
        parameters['axis_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.positionVectors[idx1,:])

        parameters['q_init'] = normalize(q_mult(self.positionVectors[idx2,3:], q_inv(self.positionVectors[idx1,3:])))

        self.addConstraint("limit", [object1, object2], parameters)


    def addSensor(self, object, reference_object):
        self.sensors.append((object,reference_object))

    def compile(self):

        self.rot_matrices = np.zeros((len(self.objects),3,3))
        for i in xrange(len(self.objects)):
            self.rot_matrices[i,:,:] = quat_to_rot_matrix(self.positionVectors[i,3:])

        self.inertia_inv = np.linalg.inv(self.massMatrices)
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

        self.P = np.zeros((self.num_constraints,))# constant
        self.w = np.zeros((self.num_constraints,))  # 0 constraints
        self.zeta = np.zeros((self.num_constraints,))  # 0 constraints

        self.only_when_positive = np.zeros((self.num_constraints,))  # 0 constraints
        self.map_object_to_constraint = [[] for _ in xrange(len(self.objects))]
        self.clipping_a   = np.ones((self.num_constraints,))
        self.clipping_idx = range(self.num_constraints)
        self.clipping_b   = np.zeros((self.num_constraints,))

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            idx2 = references[1]

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                for i in xrange(3):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 3

            if constraint == "slider" or constraint == "fixed":

                for i in xrange(3):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 3


            if constraint == "hinge":
                for i in xrange(2):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.map_object_to_constraint[idx2].append(2*(c_idx+i) + 1)

                self.w[c_idx:c_idx+3] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+3] = parameters["zeta"]
                c_idx += 2

            if constraint == "limit":
                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.map_object_to_constraint[idx2].append(2*c_idx + 1)

                self.only_when_positive[c_idx] = 1.0
                self.w[c_idx] = parameters["f"] * 2*np.pi
                self.zeta[c_idx] = parameters["zeta"]
                c_idx += 1

            if constraint == "motor":
                self.map_object_to_constraint[idx1].append(2*c_idx + 0)
                self.map_object_to_constraint[idx2].append(2*c_idx + 1)

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

            if constraint == "ground" and parameters["mu"]!=0:
                for i in xrange(2):
                    self.clipping_a[c_idx+i] = parameters["mu"]
                    self.clipping_idx[c_idx+i] = ground_contact_idx
                    self.clipping_b[c_idx+i] = 0
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.w[c_idx+i] = parameters["f"] * 2*np.pi
                    self.zeta[c_idx+i] = parameters["zeta"]

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

                c_idx += 1



    def evaluate(self, dt, positions, velocities, motor_signals):
        # TODO: drop quaternions, use rotation matrices!

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!
        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,0,0,0,0])  # total force acting on body outside of constraints
        acceleration = np.array([0,0,-9.81,0,0,0])  # acceleration of the default frame
        newv = velocities + dt * (np.dot(self.massMatrices, totalforce) + acceleration[None,:])
        originalv = newv.copy()


        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        # convert mass matrices to world coordinates
        M = np.zeros_like(self.inertia_inv)
        for i in xrange(self.inertia_inv.shape[0]):
            M[i,:3,:3] = self.inertia_inv[i,:3,:3]
            M[i,3:,3:] = np.dot(np.dot(
                self.rot_matrices[i,:,:].T,
                self.inertia_inv[i,3:,3:]
            ),
                self.rot_matrices[i,:,:]
            )

        #self.P = np.zeros((self.num_constraints,))# constant
        # changes every timestep
        J = np.zeros((self.num_constraints,2,6))  # 0 constraints x 2 objects x 6 states
        b_res = np.zeros((self.num_constraints,))  # 0 constraints
        b_error = np.zeros((self.num_constraints,))  # 0 constraints
        self.C = np.ones((self.num_constraints,))  # 0 constraints
        mass_matrix = np.zeros((self.num_constraints,2,6,6))  # 0 constraints x 2 objects x 6 states x 6 states

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            idx2 = references[1]

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                r1x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"], self.rot_matrices[idx1,:,:])
                r2x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"], self.rot_matrices[idx2,:,:])
                ss_r1x = skew_symmetric(r1x)
                ss_r2x = skew_symmetric(r2x)
                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([-np.eye(3), ss_r1x])[:,i]
                    J[c_idx+i,1,:] = np.concatenate([ np.eye(3),-ss_r2x])[:,i]
                    mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                    mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]

                b_error[c_idx:c_idx+3] = (positions[idx2,:3]+r2x-positions[idx1,:3]-r1x)
                c_idx += 3

            if constraint == "slider" or constraint == "fixed":
                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([np.zeros((3,3)),-np.eye(3)])[:,i]
                    J[c_idx+i,1,:] = np.concatenate([np.zeros((3,3)), np.eye(3)])[:,i]
                    mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                    mass_matrix[c_idx+i,1,:,:] = M[idx2,:,:]

                q_current = q_div(positions[idx2,3:], positions[idx1,3:])
                q_diff = q_div(q_current, parameters['q_init'])
                b_error[c_idx:c_idx+3] = 2*q_diff[1:]
                c_idx += 3

            if constraint == "hinge":
                a2x = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], self.rot_matrices[idx2,:,:])
                b1x = convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                c1x = convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                ss_a2x = skew_symmetric(a2x)

                J[c_idx+0,0,:] = np.concatenate([np.zeros((3,)),-np.dot(b1x,ss_a2x)])
                J[c_idx+0,1,:] = np.concatenate([np.zeros((3,)), np.dot(b1x,ss_a2x)])
                J[c_idx+1,0,:] = np.concatenate([np.zeros((3,)),-np.dot(c1x,ss_a2x)])
                J[c_idx+1,1,:] = np.concatenate([np.zeros((3,)), np.dot(c1x,ss_a2x)])
                mass_matrix[c_idx+0,0,:,:] = M[idx1,:,:]
                mass_matrix[c_idx+0,1,:,:] = M[idx2,:,:]
                mass_matrix[c_idx+1,0,:,:] = M[idx1,:,:]
                mass_matrix[c_idx+1,1,:,:] = M[idx2,:,:]

                b_error[c_idx:c_idx+2] = np.array([np.sum(a2x*b1x),np.sum(a2x*c1x)])
                c_idx += 2

            if constraint == "limit":
                angle = parameters["angle"]/180. * np.pi
                q_current = q_mult(positions[idx2,3:], q_inv(positions[idx1,3:]))
                q_diff = q_mult(q_current, q_inv(parameters['q_init']))
                a = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                dot = np.sum(q_diff[1:] * a)
                sin_theta2 = ((dot>0) * 2 - 1) * np.sqrt(np.sum(q_diff[1:]*q_diff[1:]))
                theta = 2*np.arctan2(sin_theta2,q_diff[0])


                rot_current = np.dot(self.rot_matrices[idx1,:,:], self.rot_matrices[idx2,:,:].T)
                rot_diff = np.dot(rot_current, quat_to_rot_matrix(parameters['q_init']).T)
                theta2 = np.arccos(0.5*(np.trace(rot_diff)-1))

                dot2 = np.sum(np.dot(rot_diff, a))

                print theta,theta2,"->",dot,dot2

                if parameters["angle"] < 0:
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,)),-a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,)), a])
                else:
                    J[c_idx,0,:] = np.concatenate([np.zeros((3,)), a])
                    J[c_idx,1,:] = np.concatenate([np.zeros((3,)),-a])

                mass_matrix[c_idx,0,:,:] = M[idx1,:,:]
                mass_matrix[c_idx,1,:,:] = M[idx2,:,:]

                b_error[c_idx] = np.abs(angle - theta)
                if parameters["angle"] > 0:
                    b_error[c_idx] = angle - theta
                    self.C[c_idx] = (theta > angle)
                else:
                    b_error[c_idx] = theta - angle
                    self.C[c_idx] = (theta < angle)

                c_idx += 1

            if constraint == "motor":

                a = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                q_current = q_div(positions[idx2,3:], positions[idx1,3:])
                q_diff = q_div(q_current, parameters['q_init'])
                dot = np.sum(q_diff[1:] * a)
                sin_theta2 = np.sqrt(np.sum(q_diff[1:]*q_diff[1:]))
                theta = 2*((dot>0) * 2 - 1)*np.arctan2(sin_theta2,q_diff[0])

                J[c_idx,0,:] = np.concatenate([np.zeros((3,)),-a])
                J[c_idx,1,:] = np.concatenate([np.zeros((3,)), a])
                mass_matrix[c_idx,0,:,:] = M[idx1,:,:]
                mass_matrix[c_idx,1,:,:] = M[idx2,:,:]

                motor_signal = motor_signals[parameters["motor_id"]]
                motor_signal = np.clip(motor_signal, parameters["min"]/180. * np.pi, parameters["max"]/180. * np.pi)

                if parameters["type"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["type"] == "position":
                    if "delta" in parameters:
                        b_error[c_idx] = (abs(theta-motor_signal) > parameters["delta"]) * (2*(theta>motor_signal)-1) * parameters["motor_velocity"]
                    else:
                        b_error[c_idx] = (2*(theta>motor_signal)-1) * parameters["motor_velocity"]
                c_idx += 1

            if constraint == "ground":
                r = self.radii[idx1]
                J[c_idx,0,:] = np.array([0,0,1,0,0,0])
                J[c_idx,1,:] = np.array([0,0,0,0,0,0])
                mass_matrix[c_idx,0,:,:] = M[idx1,:,:]

                b_error[c_idx] = np.max(positions[idx1,Z] - r - parameters["delta"],0)
                b_res[c_idx] = parameters["alpha"] * newv[idx1,Z]
                self.C[c_idx] = (positions[idx1,Z] - r < 0.0)
                c_idx += 1

            if constraint == "ground" and parameters["mu"]!=0:
                r = self.radii[idx1]
                for i in xrange(2):
                    if i==0:
                        J[c_idx+i,0,:] = np.array([0,1,0,-r,0,0])
                    else:
                        J[c_idx+i,0,:] = np.array([1,0,0,0,r,0])
                    J[c_idx+i,1,:] = np.array([0,0,0,0,0,0])
                    self.C[c_idx+i] = (positions[idx1,Z] - r < 0.0)
                    mass_matrix[c_idx+i,0,:,:] = M[idx1,:,:]
                c_idx += 2

            if constraint == "ground" and parameters["torsional_friction"]:
                r = self.radii[idx1]
                J[c_idx,0,:] = np.array([0,0,0,0,0,r])
                J[c_idx,1,:] = np.array([0,0,0,0,0,0])
                self.C[c_idx] = (positions[idx1,Z] < 0.0)
                b_res[c_idx] = 0
                mass_matrix[c_idx,0,:,:] = M[idx1,:,:]
                c_idx += 1

        v = np.zeros((self.num_constraints,2,6))  # 0 constraints x 2 objects x 6 states

        for iteration in xrange(self.num_iterations):
            # changes every iteration
            c_idx = 0
            for constraint,references,parameters in self.constraints:
                idx1 = references[0]
                idx2 = references[1]

                if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                    for i in xrange(3):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                    c_idx += 3

                if constraint == "slider" or constraint == "fixed":
                    for i in xrange(3):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                    c_idx += 3

                if constraint == "hinge":
                    for i in xrange(2):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        v[c_idx+i,1,:] = newv[idx2,:]
                    c_idx += 2

                if constraint == "limit":
                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = newv[idx2,:]
                    c_idx += 1

                if constraint == "motor":
                    v[c_idx,0,:] = newv[idx1,:]
                    v[c_idx,1,:] = newv[idx2,:]
                    c_idx += 1

                if constraint == "ground":
                    v[c_idx,0,:] = newv[idx1,:]
                    #v[c_idx,1,:] = np.array([0,0,0,0,0,0])
                    c_idx += 1

                if constraint == "ground" and parameters["mu"]!=0:
                    for i in xrange(2):
                        v[c_idx+i,0,:] = newv[idx1,:]
                        #v[c_idx+i,1,:] = np.array([0,0,0,0,0,0])
                    c_idx += 2

                if constraint == "ground" and parameters["torsional_friction"]:
                    v[c_idx,0,:] = newv[idx1,:]
                    #v[c_idx,1,:] = np.array([0,0,0,0,0,0])
                    c_idx += 1

            m_eff = 1./np.sum(np.sum(J[:,:,None,:]*mass_matrix, axis=-1)*J, axis=(-1,-2))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)
            ERP = dt*k/(c+dt*k)

            m_c = 1./(1./m_eff + CFM)
            b = ERP/dt * b_error + b_res
            lamb = - m_c * (np.sum(J*v, axis=(-1,-2)) + CFM * self.P + b)

            self.P += lamb
            #print np.sum(lamb**2), np.sum(self.P**2)
            clipping_limit = np.abs(self.clipping_a * self.P[self.clipping_idx] + self.clipping_b)
            self.P = np.clip(self.P,-clipping_limit, clipping_limit)
            applicable = (1-(self.only_when_positive*( 1-(self.P>=0)) )) * self.C

            result = np.sum(mass_matrix*J[:,:,None,:], axis=(-1)) * self.P[:,None,None] * applicable[:,None,None]
            result = result.reshape(result.shape[:-3] + (2*self.num_constraints,6))
            for i in xrange(newv.shape[0]):
                newv[i,:] = originalv[i,:] + np.sum(result[self.map_object_to_constraint[i],:], axis=0)

        #print
        return newv


    def do_time_step(self, dt=1e-3, motor_signals=list()):
        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.
        self.velocityVectors = self.evaluate(dt, self.positionVectors, self.velocityVectors, motor_signals=motor_signals)

        self.positionVectors[:,:3] = self.positionVectors[:,:3] + self.velocityVectors[:,:3] * dt

        v_norm = np.linalg.norm(self.velocityVectors[:,3:], axis=-1)
        a = self.velocityVectors[:,3:] / (v_norm + 1e-13)[:,None]
        theta = v_norm*dt
        self.positionVectors[:,3:] = normalize(q_mult(self.positionVectors[:,3:].T, [np.cos(theta/2), a[:,0]*np.sin(theta/2), a[:,1]*np.sin(theta/2), a[:,2]*np.sin(theta/2)]))

        for i in xrange(len(self.objects)):
            self.rot_matrices[i,:,:] = quat_to_rot_matrix(self.positionVectors[i,3:])

    def getPosition(self, reference):
        idx = self.objects[reference]
        return self.positionVectors[idx]

    def getSensorValues(self, reference):
        # make positionvectors neutral according to reference object
        idx = self.objects[reference]
        rot_matrix = self.rot_matrices[idx]

        ref_position = self.positionVectors[idx,:]

        result = self.positionVectors
        result[:,:2] = result[:,:2]-ref_position[None,:2]
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
    res = np.array([w, x, y, z]).T
    return res

def q_inv(q):
    w, x, y, z = q
    return [w,-x,-y,-z]


def q_div(q1, q2):
    w, x, y, z = q2
    return q_mult([w,-x,-y,-z], q1)

def normalize(q):
    return q/np.linalg.norm(q, axis=-1, keepdims=True)
