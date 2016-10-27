import json
from math import pi
from TheanoPhysicsSystem import theano_to_print

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
    return theano_dot_last_dimension_vector_matrix(coor, rot_matrix)
    #return T.sum(rot_matrix * coor[:,:,None], axis=1)

def theano_dot_last_dimension_matrices(x, y):
    if x.ndim==3 and y.ndim==3:
        if ("theano" in str(type(x)) and x.broadcastable[0] == False)\
                or ("numpy" in str(type(x)) and x.shape[0] != 1):
            return T.batched_dot(x, y)
        else:
            return T.tensordot(x[0,:,:], y, axes=[[1], [1]])
    else:
        return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)])

def theano_dot_last_dimension_vectors(x, y):
    return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-1,)])

def theano_dot_last_dimension_vector_matrix(x, y):
    if x.ndim==2 and y.ndim==3:
        if ("theano" in str(type(x)) and x.broadcastable[0] == False)\
                or ("numpy" in str(type(x)) and x.shape[0] != 1):
            return T.batched_dot(x[:,None,:], y)[:,0,:]
        else:
            return T.tensordot(x[0,:], y, axes=[[0], [1]])
    else:
        return T.batched_tensordot(x, y, axes=[(x.ndim-1,),(y.ndim-2,)])

def theano_dot_last_dimension_matrix_vector(x, y):
    return T.batched_tensordot(x, y, axes=[(x.ndim-1,), (y.ndim-1,)])


def theano_stack_batched_integers_mixed_numpy(L, expected_shape=None):
    """
    It is more efficient to stack in numpy than in theano. So stack numpy's first, than stack them all at once with Theano.
    """
    r = []
    last_numpy = None
    for l in L:
        if "theano" in str(type(l)):
            if last_numpy is not None:
                r.append(last_numpy.astype('float32'))
                last_numpy = None
            if expected_shape is None:
                l = l.dimshuffle(0,'x')
            r.append(l)
        else:
            if expected_shape is None:
                l = l[:,None]
            if last_numpy is None:
                last_numpy = l
            else:
                last_numpy = np.concatenate([last_numpy, l], axis=(1 if expected_shape is None else 0))
    if last_numpy is not None:
        r.append(last_numpy.astype('float32'))

    if expected_shape is None:
        result = T.concatenate(r, axis=1)
    else:
        result = T.concatenate(r, axis=0).reshape(expected_shape)
    return result

def numpy_skew_symmetric(x):
    a,b,c = x[...,0,None,None],x[...,1,None,None],x[...,2,None,None]
    z = np.zeros(x.shape[:-1]+(1,1))
    return np.concatenate([
                    np.concatenate(( z,-c, b),axis=-1),
                    np.concatenate(( c, z,-a),axis=-1),
                    np.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)

def single_skew_symmetric(x):
    a,b,c = x[:,0,None,None],x[:,1,None,None],x[:,2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)

def batch_skew_symmetric(x):
    a,b,c = x[:,:,0,None,None],x[:,:,1,None,None],x[:,:,2,None,None]
    z = T.zeros_like(a)
    return T.concatenate([
                    T.concatenate(( z,-c, b),axis=-1),
                    T.concatenate(( c, z,-a),axis=-1),
                    T.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)


def numpy_repeat_new_axis(x, times):
    return np.tile(x[None,...], [times] + x.ndim * [1])


class BatchedTheanoRigid3DBodyEngine(object):
    def __init__(self):
        self.radii = np.zeros(shape=(0,), dtype='float32')
        self.positionVectors = np.zeros(shape=(0,3), dtype='float32')
        self.rot_matrices = np.zeros(shape=(0,3,3), dtype='float32')
        self.velocityVectors = np.zeros(shape=(0,6), dtype='float32')
        self.massMatrices = np.zeros(shape=(0,6,6), dtype='float32')
        self.objects = dict()
        self.constraints = []
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
        self.lower_inertia_inv = None
        self.upper_inertia_inv = None
        self.batch_size = None
        self.num_bodies = None
        self.num_sensors = None

        self.DT = None
        self.projected_gauss_seidel_iterations = None
        self.rotation_reorthogonalization_iterations = None
        self.warm_start = None

    def set_integration_parameters(self,
                                   time_step=0.001,
                                   projected_gauss_seidel_iterations=1,
                                   rotation_reorthogonalization_iterations=1,
                                   warm_start=0):
        self.DT = time_step
        self.projected_gauss_seidel_iterations = projected_gauss_seidel_iterations
        self.rotation_reorthogonalization_iterations = rotation_reorthogonalization_iterations
        self.warm_start = warm_start

    def addCube(self, reference, dimensions, mass_density, position, rotation, velocity, **kwargs):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, -1)
        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype='float32'), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype='float32'), axis=0)
        mass = mass_density*np.prod(dimensions)
        I1 = 1./12. * (dimensions[1]**2 + dimensions[2]**2)
        I2 = 1./12. * (dimensions[0]**2 + dimensions[2]**2)
        I3 = 1./12. * (dimensions[0]**2 + dimensions[1]**2)
        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,I1,I2,I3])[None,:,:], axis=0)


    def addSphere(self, reference, radius, mass_density, position, rotation, velocity, **kwargs):
        self.objects[reference] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, radius)
        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype='float32'), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype='float32'), axis=0)
        mass = mass_density*4./3.*np.pi*radius**3

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


    def load_robot_model(self, filename):
        robot_dict = json.load(open(filename,"rb"))

        self.set_integration_parameters(**robot_dict["integration_parameters"])

        for elementname, element in robot_dict["model"].iteritems():
            primitive = element[0]
            parameters = dict(robot_dict["default_model_parameters"]["default"])  # copy
            if primitive["shape"] in robot_dict["default_model_parameters"]:
                parameters.update(robot_dict["default_model_parameters"][primitive["shape"]])
            parameters.update(primitive)
            if primitive["shape"] == "cube":
                self.addCube(elementname, **parameters)
            elif primitive["shape"] == "sphere":
                self.addSphere(elementname, **parameters)

        for jointname, joint in robot_dict["joints"].iteritems():
            parameters = dict(robot_dict["default_constraint_parameters"]["default"])  # copy
            if joint["type"] in robot_dict["default_constraint_parameters"]:
                parameters.update(robot_dict["default_constraint_parameters"][joint["type"]])
            parameters.update(joint)
            if joint["type"] == "hinge":
                self.addHingeConstraint(jointname, **parameters)

            elif joint["type"] == "ground":
                self.addGroundConstraint(jointname, **parameters)

            elif joint["type"] == "fixed":
                self.addFixedConstraint(jointname, **parameters)

            elif joint["type"] == "ball":
                self.addBallAndSocketConstraint(jointname, **parameters)

            if "limits" in parameters:
                for limit in parameters["limits"]:
                    limitparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "limit" in robot_dict["default_constraint_parameters"]:
                        limitparameters.update(robot_dict["default_constraint_parameters"]["limit"])
                    limitparameters.update(limit)
                    self.addLimitConstraint(joint["object1"], joint["object2"], **limitparameters)

            if "motors" in parameters:
                for motor in parameters["motors"]:
                    motorparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "motor" in robot_dict["default_constraint_parameters"]:
                        motorparameters.update(robot_dict["default_constraint_parameters"]["motor"])
                    motorparameters.update(motor)
                    self.addMotorConstraint(joint["object1"], joint["object2"], **motorparameters)

        for sensor in robot_dict["sensors"]:
            self.addSensor(**sensor)

    def compile(self, batch_size=1):
        self.batch_size = batch_size
        self.num_bodies = self.positionVectors.shape[0]
        self.num_sensors = len(self.sensors)

        self.num_constraints = 0

        for (constraint,references,parameters) in self.constraints:
            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                self.num_constraints += 3
            if constraint == "slider" or constraint == "fixed":
                self.num_constraints += 3
            if constraint == "hinge":
                self.num_constraints += 2
            if constraint == "ground":
                self.num_constraints += 1
            if constraint == "ground" and parameters["mu"]!=0:
                self.num_constraints += 2
            if constraint == "ground" and parameters["torsional_friction"] and parameters["mu"]!=0:
                self.num_constraints += 1
            if constraint == "limit":
                self.num_constraints += 1
            if constraint == "motor":
                self.num_constraints += 1

        self.P = T.zeros((self.batch_size, self.num_constraints,))
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
                self.zero_index.append(idx1)
                self.one_index.append(idx2)
                c_idx += 1

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

            if constraint == "ground" and parameters["torsional_friction"] and parameters["mu"]!=0:
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

        self.positionVectors = theano.shared(numpy_repeat_new_axis(self.positionVectors, self.batch_size).astype('float32'), name="positionVectors")
        self.velocityVectors = theano.shared(numpy_repeat_new_axis(self.velocityVectors, self.batch_size).astype('float32'), name="velocityVectors")
        self.rot_matrices = theano.shared(numpy_repeat_new_axis(self.rot_matrices, self.batch_size).astype('float32'), name="rot_matrices", )
        self.lower_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,:3,:3]), self.batch_size).astype('float32'), name="lower_inertia_inv", )
        self.upper_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,3:,3:]), self.batch_size).astype('float32'), name="upper_inertia_inv", )


    def getSharedVariables(self):
        return [
            self.positionVectors,
            self.velocityVectors,
            self.rot_matrices,
            self.lower_inertia_inv,
            self.upper_inertia_inv,
        ]


    def evaluate(self, dt, positions, velocities, rot_matrices, motor_signals):

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!
        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,0,0,0,0], dtype='float32')  # total force acting on body outside of constraints
        acceleration = np.array([0,0,-9.81,0,0,0], dtype='float32')  # acceleration of the default frame
        newv = velocities + dt * acceleration[None,None,:]
        originalv = newv


        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        # convert mass matrices to world coordinates
        M = T.zeros(shape=(self.batch_size,self.num_bodies,6,6))


        M00 = self.lower_inertia_inv
        M01 = T.zeros(shape=(self.batch_size,self.num_bodies,3,3))
        M10 = M01
        M11 = T.sum(rot_matrices[:,:,:,None,:,None] * self.upper_inertia_inv[:,:,:,:,None,None] * rot_matrices[:,:,None,:,None,:], axis=(2,3))
        M0 = T.concatenate([M00,M01],axis=3)
        M1 = T.concatenate([M10,M11],axis=3)
        M = T.concatenate([M0,M1],axis=2)

        # changes every timestep

        # constraints are first dimension! We will need to stack them afterwards!
        J = [np.zeros((self.batch_size,6), dtype="float32") for _ in xrange(2 * self.num_constraints)]   # 0 constraints x 0 bodies x 2 objects x 6 states
        b_res =   [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies
        b_error = [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies
        C =       [np.zeros((self.batch_size,), dtype="float32") for _ in xrange(self.num_constraints)]  # 0 constraints x 0 bodies

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            if references[1] is not None:
                idx2 = references[1]
            else:
                idx2 = None

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                r1x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"][None,:], rot_matrices[:,idx1,:,:])
                r2x = theano_convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"][None,:], rot_matrices[:,idx2,:,:])
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

            if constraint == "slider" or constraint == "fixed":
                batched_eye = numpy_repeat_new_axis(np.eye(3, dtype='float32'), self.batch_size)
                batched_zeros = numpy_repeat_new_axis(np.zeros((3,3), dtype='float32'), self.batch_size)


                complete_J1 = np.concatenate([batched_zeros,-batched_eye],axis=1)
                complete_J2 = np.concatenate([batched_zeros, batched_eye],axis=1)

                for i in xrange(3):
                    J[2*(c_idx+i)+0] = complete_J1[:,:,i]
                    J[2*(c_idx+i)+1] = complete_J2[:,:,i]

                rot_current = np.dot(rot_matrices[idx2,:,:], rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, parameters['rot_init'].T)
                cross = rot_diff.T - rot_diff
                # TODO: add stabilization of this constraint
                b_error[c_idx] = np.zeros(shape=(self.batch_size,))#cross[1,2]
                b_error[c_idx+1] = np.zeros(shape=(self.batch_size,))#cross[2,0]
                b_error[c_idx+2] = np.zeros(shape=(self.batch_size,))#cross[0,1]
                c_idx += 3

            if constraint == "hinge":
                a2x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'][None,:], rot_matrices[:,idx2,:,:])
                b1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'][None,:], rot_matrices[:,idx1,:,:])
                c1x = theano_convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'][None,:], rot_matrices[:,idx1,:,:])
                ss_a2x = single_skew_symmetric(a2x)

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)

                J[2*(c_idx+0)+0] = T.concatenate([batched_zeros,-theano_dot_last_dimension_vector_matrix(b1x,ss_a2x)],axis=1)
                J[2*(c_idx+0)+1] = T.concatenate([batched_zeros, theano_dot_last_dimension_vector_matrix(b1x,ss_a2x)],axis=1)
                J[2*(c_idx+1)+0] = T.concatenate([batched_zeros,-theano_dot_last_dimension_vector_matrix(c1x,ss_a2x)],axis=1)
                J[2*(c_idx+1)+1] = T.concatenate([batched_zeros, theano_dot_last_dimension_vector_matrix(c1x,ss_a2x)],axis=1)

                b_error[c_idx+0] = theano_dot_last_dimension_vectors(a2x,b1x)
                b_error[c_idx+1] = theano_dot_last_dimension_vectors(a2x,c1x)
                c_idx += 2

            """
            if constraint == "limit":
                #TODO: move to Theano...
                angle = parameters["angle"]/180. * np.pi
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], rot_matrices[:,idx1,:,:])
                rot_current = theano_dot_last_dimension_matrices(rot_matrices[:,idx2,:,:], rot_matrices[idx1,:,:].dimshuffle(0,2,1))
                rot_diff = theano_dot_last_dimension_matrices(rot_current, quat_to_rot_matrix(parameters['rot_init']).dimshuffle(0,2,1))
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
            """
            if constraint == "motor":
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'][None,:], rot_matrices[:,idx1,:,:])

                # TODO: remove dimshuffle(0,2,1) by using batched_dot
                rot_current = theano_dot_last_dimension_matrices(rot_matrices[:,idx2,:,:], rot_matrices[:,idx1,:,:].dimshuffle(0,2,1))
                rot_init = numpy_repeat_new_axis(parameters['rot_init'].T, self.batch_size)
                rot_diff = theano_dot_last_dimension_matrices(rot_current, rot_init)

                traces = rot_diff[:,0,0] + rot_diff[:,1,1] + rot_diff[:,2,2]
                #traces =  theano.scan(lambda y: T.nlinalg.trace(y), sequences=rot_diff)[0]

                theta2 = T.arccos(T.clip(0.5*(traces-1),-1+eps,1-eps))
                cross = rot_diff.dimshuffle(0,2,1) - rot_diff
                dot2 = cross[:,1,2] * a[:,0] + cross[:,2,0] * a[:,1] + cross[:,0,1] * a[:,2]

                theta = ((dot2>0) * 2 - 1) * theta2

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)
                J[2*c_idx+0] = T.concatenate([batched_zeros,-a],axis=1)
                J[2*c_idx+1] = T.concatenate([batched_zeros, a],axis=1)

                motor_signal = motor_signals[:,parameters["motor_id"]]

                motor_min = (parameters["min"]/180. * np.pi)
                motor_max = (parameters["max"]/180. * np.pi)
                motor_signal = T.clip(motor_signal, motor_min, motor_max).astype('float32')

                if parameters["type"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["type"] == "position":
                    if "delta" in parameters and parameters["delta"]>0:
                        b_error[c_idx] = dt * (abs(theta-motor_signal) > parameters["delta"]) * (theta-motor_signal) * parameters["motor_gain"]
                    else:
                        b_error[c_idx] = dt * (theta-motor_signal) * parameters["motor_gain"]

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

        #mass_matrix = T.concatenate((M[:,self.zero_index,None,:,:], M[:,self.one_index,None,:,:]), axis=2)
        #mass_matrix = T.concatenate((
        #                            T.stack([M[:,i,None,:,:] for i in self.zero_index], axis=1),
        #                            T.stack([M[:,j,None,:,:] for j in self.one_index], axis=1)
        #                            ), axis=2)

        #J = T.stack(J, axis=0).reshape(shape=(self.num_constraints,2,self.batch_size,6)).dimshuffle(2,0,1,3)

        J = theano_stack_batched_integers_mixed_numpy(J, expected_shape=(self.num_constraints,2,self.batch_size,6)).dimshuffle(2,0,1,3)

        C = theano_stack_batched_integers_mixed_numpy(C, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)
        b_res = theano_stack_batched_integers_mixed_numpy(b_res, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)
        b_error = theano_stack_batched_integers_mixed_numpy(b_error, expected_shape=(self.num_constraints,self.batch_size)).dimshuffle(1,0)

        #v = np.zeros((self.num_constraints,2,6))  # 0 constraints x 2 objects x 6 states

        self.P = self.warm_start * self.P

        for iteration in xrange(self.projected_gauss_seidel_iterations):
            # changes every iteration
            v = newv[:,zipped_indices,:].reshape(shape=(self.batch_size, self.num_constraints, 2, 6))
            #v = T.concatenate((newv[self.zero_index,None,:],newv[self.one_index,None,:]),axis=1)
            #v = T.concatenate((
            #    T.stack([newv[:,i,None,:] for i in self.zero_index], axis=1),
            #    T.stack([newv[:,j,None,:] for j in self.one_index], axis=1)
            #),axis=2)

            # TODO: batch-dot-product
            m_eff = 1./T.sum(T.sum(J[:,:,:,None,:]*mass_matrix, axis=4)*J, axis=(2,3))
            #m_eff = 1./T.sum(T.sum(J[:,:,None,:]*mass_matrix, axis=-1)*J, axis=(-1,-2))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)
            ERP = dt*k/(c+dt*k)

            m_c = 1./(1./m_eff + CFM)

            b = ERP/dt * b_error + b_res

            lamb = - m_c * (T.sum(J*v, axis=(2,3)) + CFM * self.P + b)

            self.P += lamb
            #print J[[39,61,65,69],:]
            #print np.sum(lamb**2), np.sum(self.P**2)

            #clipping_force = T.concatenate([self.P[:,j].dimshuffle(0,'x') for j in self.clipping_idx],axis=1)
            clipping_force = self.P[:,self.clipping_idx]

            clipping_limit = abs(self.clipping_a * clipping_force + self.clipping_b * dt)
            self.P = T.clip(self.P,-clipping_limit, clipping_limit)
            applicable = (1.0*(C<=0)) * (1-(self.only_when_positive*(self.P<=0)))

            # TODO: batch-dot-product
            result = T.sum(mass_matrix*J[:,:,:,None,:], axis=4) * (self.P * applicable)[:,:,None,None]
            #result = theano_dot_last_dimension_matrix_vector(mass_matrix,J) * (self.P * applicable)[:,:,None,None]

            result = result.reshape((self.batch_size, 2*self.num_constraints, 6))

            r = []
            for i in xrange(len(self.map_object_to_constraint)):
                delta_v = T.sum(result[:,self.map_object_to_constraint[i],:], axis=1)
                r.append(delta_v)
            newv = newv + T.stack(r, axis=1)
        #print
        return newv


    def temp(self):

        #for iteration in xrange(self.projected_gauss_seidel_iterations):
        """
        def projected_gauss_seidel(loopv, P):
            # changes every iteration
            #v = T.concatenate((newv[self.zero_index,None,:],newv[self.one_index,None,:]),axis=1)
            v = T.concatenate((
                T.stack([loopv[:,i,None,:] for i in self.zero_index], axis=1),
                T.stack([loopv[:,j,None,:] for j in self.one_index], axis=1)
            ),axis=2)

            # TODO: batch-dot-product
            m_eff = 1./T.sum(T.sum(J[:,:,:,None,:]*mass_matrix, axis=4)*J, axis=(2,3))
            #m_eff = 1./T.sum(T.sum(J[:,:,None,:]*mass_matrix, axis=-1)*J, axis=(-1,-2))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)
            ERP = dt*k/(c+dt*k)

            m_c = 1./(1./m_eff + CFM)

            b = ERP/dt * b_error + b_res
            # TODO: batch-dot-product
            lamb = - m_c * (T.sum(J*v, axis=(2,3)) + CFM * P + b)

            P += lamb

            clipping_force = T.concatenate([P[:,j].dimshuffle(0,'x') for j in self.clipping_idx],axis=1)
            clipping_limit = abs(self.clipping_a * clipping_force + self.clipping_b * dt)
            P = T.clip(P,-clipping_limit, clipping_limit)
            applicable = (1-(self.only_when_positive*( 1-(P>=0)) )) * (C<=0)

            # TODO: batch-dot-product
            result = T.sum(mass_matrix*J[:,:,:,None,:], axis=4) * P[:,:,None,None] * applicable[:,:,None,None]
            result = result.reshape((self.batch_size, 2*self.num_constraints, 6))

            r = []
            for i in xrange(len(self.map_object_to_constraint)):
                # TODO: remove the stacking?
                delta_v = T.sum(T.stack([result[:,j,:] for j in self.map_object_to_constraint[i]],axis=1), axis=1)
                r.append(delta_v)
            loopv = loopv + T.stack(r, axis=1)

            return loopv, P

        print "ELLO", newv.ndim, self.P.ndim
        outputs, updates = theano.scan(
            fn=lambda nv, pp: projected_gauss_seidel(nv, pp),
            outputs_info=(newv,self.P),
            n_steps=self.projected_gauss_seidel_iterations
        )
        assert len(updates)==0
        newv = outputs[0][-1]
        self.P = outputs[1][-1]
        print "ELLO", newv.ndim, self.P.ndim
        #print
        return newv
        """

    def step_from_this_state(self, state, dt=None, motor_signals=list()):
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

        return (positions, velocities, rot_matrices)


    def normalize_matrix(self, A):
        for i in xrange(self.rotation_reorthogonalization_iterations):
            # TODO: batch-dot-product
            A = (3*A - T.sum(A[:,:,:,None,:,None] * A[:,:,None,:,:,None] * A[:,:,None,:,None,:], axis=(3,4))) / 2
        return A



    def getInitialState(self):
        return self.positionVectors, self.velocityVectors, self.rot_matrices

    def randomizeInitialState(self, rotate_around):
        srng = RandomStreams(seed=317070)
        # step 1, rotate the robots slightly. 360 degrees around z-axis. 5 degrees x- and y-axis.
        # rotate around
        # TODO: rotate velocities
        """
        rot_index = self.getObjectIndex(rotate_around)
        rot_point = self.positionVectors[:,rot_index,:]
        r_x = srng.uniform(size=(self.batch_size,1), low=-0.05, high=0.05)
        r_y = srng.uniform(size=(self.batch_size,1), low=-0.05, high=0.05)
        r_z = srng.uniform(size=(self.batch_size,1), low=0, high=2*pi)
        status = T.concatenate([self.rot_matrices, self.positionVectors[:,:,:,None]], axis=3) # (3,4) matrices
        rot_x_matrix = T.concatenate([
                    T.concatenate(( 1, 0, 0),axis=-1),
                    T.concatenate(( 0, T.cos(r_x),-T.sin(r_x)),axis=-1),
                    T.concatenate(( 0, T.sin(r_x), T.cos(r_x)),axis=-1)
                            ],axis=-2)

        rot_y_matrix = T.concatenate([
                    T.concatenate(( T.cos(r_y), 0, T.sin(r_y)),axis=-1),
                    T.concatenate(( 0, 1, 0),axis=-1),
                    T.concatenate(( -T.sin(r_y), 0, T.cos(r_y)),axis=-1)
                            ],axis=-2)

        rot_z_matrix = T.concatenate([
                    T.concatenate(( T.cos(r_z),-T.sin(r_z), 0),axis=-1),
                    T.concatenate(( T.sin(r_z), T.cos(r_z), 0),axis=-1),
                    T.concatenate(( 0, 0, 1),axis=-1)
                            ],axis=-2)

        total_rot_matrix = theano_dot_last_dimension_matrices(rot_z_matrix, theano_dot_last_dimension_matrices(rot_y_matrix, rot_x_matrix))
        """

        # this is fucking complex shit


        # step 2, move the robots in the batch to a random location
        d_x = srng.uniform(size=(self.batch_size,1), low=0., high=0)
        d_y = srng.uniform(size=(self.batch_size,1), low=0, high=0)
        d_z = srng.uniform(size=(self.batch_size,1), low=0.0, high=0.0)
        self.positionVectors = self.positionVectors + T.concatenate([d_x, d_y, d_z],axis=1)[:,None,:]
        pass

    def getPosition(self, reference):
        idx = self.getObjectIndex(reference)
        return self.positionVectors[:,idx,:]

    def getRotationMatrix(self, reference):
        idx = self.getObjectIndex(reference)
        return self.rot_matrices[:,idx,:,:]

    def getObjectIndex(self, reference):
        return self.objects[reference]

    def addSensor(self, **kwargs):
        if "axis" in kwargs and "reference" in kwargs:
            kwargs["axis"] = convert_world_to_model_coordinate_no_bias(kwargs["axis"], self.rot_matrices[self.getObjectIndex(kwargs["reference"]),:,:])
        kwargs["axis"] = np.array(kwargs["axis"], dtype='float32')
        self.sensors.append(kwargs)

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
