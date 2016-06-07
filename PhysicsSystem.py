__author__ = 'jonas'
import numpy as np

X = 0
Y = 1
Z = 2

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

        self.constraints.append([constraint, references, parameters])


    def addBallAndSocketConstraint(self, object1, object2, world_coordinates, parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = world_coordinates - self.positionVectors[idx1,:3]
        parameters['joint_in_model2_coordinates'] = world_coordinates - self.positionVectors[idx2,:3]

        self.addConstraint("ball-and-socket", [object1, object2], parameters)


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
            for constraint,references,parameters in self.constraints:

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

                    self.velocityVectors[idx] = newv[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb

                    J = np.array([[1,0,0,0,1,0]])
                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                    b = 0
                    lamb_friction_1 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]

                    lamb_friction_1 = np.clip(lamb_friction_1, -parameters["mu"]*Fn, parameters["mu"]*Fn)

                    self.velocityVectors[idx,:] = self.velocityVectors[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_1

                    J = np.array([[0,1,0,1,0,0]])
                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                    b = 0
                    lamb_friction_2 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]
                    lamb_friction_2 = np.clip(lamb_friction_2, -parameters["mu"]*Fn, parameters["mu"]*Fn)


                    self.velocityVectors[idx,:] = self.velocityVectors[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_2

                    if parameters["torsional_friction"]:
                        J = np.array([[0,0,0,0,0,1]])
                        m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))
                        b = 0
                        lamb_friction_3 = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]
                        lamb_friction_3 = np.clip(lamb_friction_3, -parameters["mu"]*Fn, parameters["mu"]*Fn)
                        self.velocityVectors[idx,:] = self.velocityVectors[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb_friction_3


                elif constraint == "ball-and-socket":
                    idx1 = references[0]
                    idx2 = references[1]

                    v = np.array([newv[idx1,:], newv[idx2,:]])
                    print v.shape

                    J = np.array([[0,0,1,0,0,0]])

                    m_c = 1./np.dot(J,np.dot(M[idx,:,:], J.T))

                    b = 0
                    lamb = (- m_c * (np.dot(J, newv[idx,:]) + b))[0,0]

                    Fn = applicable * lamb

                    self.velocityVectors[idx] = newv[idx,:] + applicable * np.dot(J, M[idx,:,:]) * lamb





        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.

        self.positionVectors[:,:3] = self.positionVectors[:,:3] + self.velocityVectors[:,:3] * dt

        v_norm = np.sqrt(np.sum(self.velocityVectors[:,3:]**2))
        a = self.velocityVectors[:,3:] / (v_norm + 1e-13)
        theta = v_norm*dt
        self.positionVectors[:,3:] = q_mult(self.positionVectors[:,3:].T, [np.cos(theta/2), a[:,0]*np.sin(theta/2), a[:,1]*np.sin(theta/2), a[:,2]*np.sin(theta/2)])


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