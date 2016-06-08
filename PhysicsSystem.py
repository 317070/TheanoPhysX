__author__ = 'jonas'
import numpy as np
import scipy.linalg

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


class Rigid3DBodyEngine(object):
    def __init__(self):
        self.positionVectors = np.zeros(shape=(0,7))
        self.velocityVectors = np.zeros(shape=(0,6))
        self.massMatrices = np.zeros(shape=(0,6,6))
        self.objects = dict()
        self.constraints = []
        self.num_iterations = 5

    def addSphere(self, reference, position, velocity):
        self.objects[reference] = self.positionVectors.shape[0]

        self.positionVectors = np.append(self.positionVectors, np.array([position]), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity]), axis=0)
        self.massMatrices = np.append(self.massMatrices, np.diag([1,1,1,0.4,0.4,0.4])[None,:,:], axis=0)

    def addConstraint(self, constraint, references, parameters):
        references = [self.objects[reference] for reference in references]

        self.constraints.append([constraint, references, parameters])


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

        parameters['axis1_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_1, self.positionVectors[idx1,:])
        parameters['axis2_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_2, self.positionVectors[idx1,:])
        parameters['axis_in_model2_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.positionVectors[idx2,:])

        self.addConstraint("hinge", [object1, object2], parameters)




    def do_time_step(self, dt=1e-3):

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!

        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,-9.81,0,0,0])  # total force acting on body outside of constraints
        newv = self.velocityVectors + dt * np.dot(self.massMatrices, totalforce)
        self.velocityVectors = newv

        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints.
        M = np.linalg.inv(self.massMatrices)

        for iteration in xrange(self.num_iterations):
            total_lambda = np.zeros_like(self.velocityVectors)
            for constraint,references,parameters in self.constraints[::-1]:

                if constraint == "ground":
                    idx = references[0]
                    applicable = (self.positionVectors[idx,Z] <= 0 and self.velocityVectors[idx,Z] <= 0)  # we cannot do advanced intersection algorithms on GPU for now

                    J = np.array([[0,0,1,0,0,0]])

                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))

                    b_error = parameters["gamma"] * np.max(self.positionVectors[idx,Z] - parameters["delta"],0)
                    b_res = parameters["alpha"] * newv[idx,Z]

                    b = b_error + b_res
                    lamb = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]

                    Fn = applicable * lamb

                    total_lambda[idx,:] = total_lambda[idx,:] +  applicable * np.dot(J, M[idx,:,:]) * lamb

                    J = np.array([[1,0,0,0,1,0]])
                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                    b = 0
                    lamb_friction_1 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]

                    lamb_friction_1 = np.clip(lamb_friction_1, -parameters["mu"]*Fn, parameters["mu"]*Fn)

                    total_lambda[idx,:] = total_lambda[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_1

                    J = np.array([[0,1,0,1,0,0]])
                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                    b = 0
                    lamb_friction_2 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]
                    lamb_friction_2 = np.clip(lamb_friction_2, -parameters["mu"]*Fn, parameters["mu"]*Fn)


                    total_lambda[idx,:] = total_lambda[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_2

                    if parameters["torsional_friction"]:
                        J = np.array([[0,0,0,0,0,1]])
                        m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                        b = 0
                        lamb_friction_3 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]
                        lamb_friction_3 = np.clip(lamb_friction_3, -parameters["mu"]*Fn, parameters["mu"]*Fn)
                        total_lambda[idx,:] = total_lambda[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_3


                if constraint == "ball-and-socket" or constraint == "hinge":
                    idx1 = references[0]
                    idx2 = references[1]

                    v = np.concatenate([newv[idx1,:], newv[idx2,:]])
                    #print v.shape
                    r1x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"], self.positionVectors[idx1,:])
                    r2x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"], self.positionVectors[idx2,:])

                    J = np.concatenate([-np.eye(3),skew_symmetric(r1x),np.eye(3),-skew_symmetric(r2x)]).T
                    mass_matrix = scipy.linalg.block_diag(M[idx1,:,:], M[idx2,:,:])
                    m_c = np.linalg.inv(np.dot(J,np.dot(mass_matrix, J.T)))

                    b_res = parameters["beta"] * (self.positionVectors[idx2,:3]+r2x-self.positionVectors[idx1,:3]-r1x)
                    #print b_res
                    b = b_res
                    lamb = - np.dot(m_c, (np.dot(J, v) + b))
                    #print v.shape, np.dot(mass_matrix, J.T).shape, lamb.shape,
                    result = np.dot(np.dot(mass_matrix, J.T), lamb)
                    #print result.shape

                    total_lambda[idx1,:] = total_lambda[idx1,:] + result[:6]
                    total_lambda[idx2,:] = total_lambda[idx2,:] + result[6:]

                if constraint == "hinge":
                    idx1 = references[0]
                    idx2 = references[1]

                    v = np.concatenate([newv[idx1,:], newv[idx2,:]])
                    #print v.shape
                    a2x = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], self.positionVectors[idx2,:])
                    b1x = convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], self.positionVectors[idx1,:])
                    c1x = convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], self.positionVectors[idx1,:])

                    #print np.dot(skew_symmetric(a2x),b1x).shape
                    J1 = np.concatenate([np.zeros((3,)),-np.dot(skew_symmetric(a2x),b1x),np.zeros((3,)),np.dot(skew_symmetric(a2x),b1x)])[:,None]
                    J2 = np.concatenate([np.zeros((3,)),-np.dot(skew_symmetric(a2x),c1x),np.zeros((3,)),np.dot(skew_symmetric(a2x),c1x)])[:,None]
                    #print J1.shape, J2.shape
                    J = np.concatenate([J1,J2],axis=1).T
                    #print J.shape

                    mass_matrix = scipy.linalg.block_diag(M[idx1,:,:], M[idx2,:,:])
                    m_c = np.linalg.inv(np.dot(J,np.dot(mass_matrix, J.T)))

                    b_res = parameters["beta"] * np.array([np.sum(a2x*b1x),np.sum(a2x*c1x)])
                    print np.array([np.sum(a2x*b1x),np.sum(a2x*c1x)])
                    b = b_res
                    lamb = - np.dot(m_c, (np.dot(J, v) + b))
                    #print v.shape, np.dot(mass_matrix, J.T).shape, lamb.shape,
                    result = np.dot(np.dot(mass_matrix, J.T), lamb)
                    #print result.shape

                    total_lambda[idx1,:] = total_lambda[idx1,:] + result[:6]
                    total_lambda[idx2,:] = total_lambda[idx2,:] + result[6:]



            self.velocityVectors += total_lambda


        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.

        self.positionVectors[:,:3] = self.positionVectors[:,:3] + self.velocityVectors[:,:3] * dt

        v_norm = np.sqrt(np.sum(self.velocityVectors[:,3:]**2))
        a = self.velocityVectors[:,3:] / (v_norm + 1e-13)
        theta = v_norm*dt
        self.positionVectors[:,3:] = (q_mult(self.positionVectors[:,3:].T, [np.cos(theta/2), a[:,0]*np.sin(theta/2), a[:,1]*np.sin(theta/2), a[:,2]*np.sin(theta/2)]))


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
    return res / np.linalg.norm(res, axis=-1, keepdims=True)
    return res

def normalize(q):
    q = np.copy(q)
    v_norm = np.linalg.norm(res, axis=-1, keepdims=True)
    q[...,1:] = q[...,1:]/v_norm[...,None]
    return q
