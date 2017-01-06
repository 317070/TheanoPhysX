import numpy as np
import theano
import theano.tensor as T
from PhysicsSystem import Rigid3DBodyEngine, EngineState
from aux import numpy_repeat_new_axis
from aux_theano import batch_skew_symmetric, theano_convert_model_to_world_coordinate_no_bias, \
    theano_stack_batched_integers_mixed_numpy, single_skew_symmetric, theano_dot_last_dimension_vector_matrix, \
    theano_dot_last_dimension_vectors, theano_dot_last_dimension_matrices, theano_convert_model_to_world_coordinate, \
    theano_convert_world_to_model_coordinate_no_bias

__author__ = 'jonas degrave'

eps=1e-4

X = 0
Y = 1
Z = 2

class TheanoRigid3DBodyEngine(Rigid3DBodyEngine):
    def __init__(self, *args, **kwargs):
        super(TheanoRigid3DBodyEngine, self).__init__(*args, **kwargs)
        self.batch_size = None
        self.lower_inertia_inv = None
        self.upper_inertia_inv = None
        self.impulses_P = None
        self.face_normal_theano = None
        self.face_point_theano = None
        self.face_texture_x_theano = None
        self.face_texture_y_theano = None

    def get_state_variables(self):
        variables = self.get_initial_state()
        res = []
        for var in variables:
            if var.ndim==0:
                res.append(T.fscalar())
            if var.ndim==1:
                res.append(T.fvector())
            if var.ndim==2:
                res.append(T.fmatrix())
            if var.ndim==3:
                res.append(T.ftensor3())
            if var.ndim==4:
                res.append(T.ftensor4())
        return EngineState(*res)

    def compile(self, batch_size, *args,**kwargs):
        super(TheanoRigid3DBodyEngine, self).compile(*args,**kwargs)

        self.batch_size = batch_size
        self.initial_positions = theano.shared(numpy_repeat_new_axis(self.initial_positions, self.batch_size).astype('float32'), name="initial_positions")
        self.initial_velocities = theano.shared(numpy_repeat_new_axis(self.initial_velocities, self.batch_size).astype('float32'), name="initial_velocities")
        self.initial_rotations = theano.shared(numpy_repeat_new_axis(self.initial_rotations, self.batch_size).astype('float32'), name="initial_rotations", )
        # For the warm start, keep the impuls of the previous timestep
        self.impulses_P = theano.shared(np.zeros(shape=(self.batch_size, self.num_constraints,), dtype='float32'), name="impulses_P")

        # TODO: make these constants instead of shared,
        # when not needed for randomization
        self.lower_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,:3,:3]), self.batch_size).astype('float32'), name="lower_inertia_inv", )
        self.upper_inertia_inv = theano.shared(numpy_repeat_new_axis(np.linalg.inv(self.massMatrices[:,3:,3:]), self.batch_size).astype('float32'), name="upper_inertia_inv", )

        self.face_normal_theano = theano.shared(numpy_repeat_new_axis(self.face_normal[:,:], self.batch_size).astype('float32'), name="face_normal")
        self.face_point_theano = theano.shared(numpy_repeat_new_axis(self.face_point[:,:], self.batch_size).astype('float32'), name="face_point")
        self.face_texture_x_theano = theano.shared(numpy_repeat_new_axis(self.face_texture_x[:,:], self.batch_size).astype('float32'), name="face_texture_x")
        self.face_texture_y_theano = theano.shared(numpy_repeat_new_axis(self.face_texture_y[:,:], self.batch_size).astype('float32'), name="face_texture_y")



    def get_shared_variables(self):
        return [
            self.initial_positions,
            self.initial_velocities,
            self.initial_rotations,
            self.lower_inertia_inv,
            self.upper_inertia_inv,
            self.impulses_P,
            self.face_normal_theano,
            self.face_point_theano,
            self.face_texture_x_theano,
            self.face_texture_y_theano,
        ]

    def do_time_step(self, state=None, dt=None, motor_signals=list()):
        if dt is None:
            dt = self.DT

        positions, velocities, rotations = state.positions, state.velocities, state.rotations
        ##################
        # --- Step 3 --- #
        ##################
        # semi-implicit Euler integration
        velocities = self.evaluate(state=state, dt=dt, motor_signals=motor_signals)

        positions = positions + velocities[:,:,:3] * dt
        # TODO: batch-dot-product
        rotations = self.normalize_matrix(rotations[:,:,:,:] + T.sum(rotations[:,:,:,:,None] * batch_skew_symmetric(dt * velocities[:,:,3:])[:,:,None,:,:],axis=3) )

        return EngineState(
            positions=positions, 
            velocities=velocities, 
            rotations=rotations)

    def normalize_matrix(self, A):
        for i in xrange(self.rotation_reorthogonalization_iterations):
            # TODO: batch-dot-product
            A = (3*A - T.sum(A[:,:,:,None,:,None] * A[:,:,None,:,:,None] * A[:,:,None,:,None,:], axis=(3,4))) / 2
        return A

    def get_initial_state(self):
        return EngineState(positions=self.initial_positions,
                           velocities=self.initial_velocities,
                           rotations=self.initial_rotations)

    def get_initial_position(self, reference):
        idx = self.get_object_index(reference)
        return self.initial_positions[:,idx,:]

    def get_initial_rotation(self, reference):
        idx = self.get_object_index(reference)
        return self.initial_rotations[:,idx,:,:]

    def get_object_index(self, reference):
        return self.objects[reference]

    def get_sensor_values(self, state):
        # make positionvectors neutral according to reference object
        positions, velocities, rot_matrices = state.positions, state.velocities, state.rotations

        r = []
        for sensor in self.sensors:
            idx = self.get_object_index(sensor["object"])
            if "reference" in sensor:
                ref_idx = self.get_object_index(sensor["reference"])
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

    def get_camera_image_size(self, camera_name):
        return (self.batch_size,) + super(TheanoRigid3DBodyEngine, self).get_camera_image_size(camera_name)

    def get_camera_image(self, state, camera_name):

        # get camera image
        # do ray-sphere and ray-plane intersections
        # find cube by using 6 planes, throwing away the irrelevant intersections

        # step 1: generate list of rays (1 per pixel)
        # focal_point (3,)
        # ray_dir (px_hor, px_ver, 3)
        # ray_offset (px_hor, px_ver, 3)

        camera = self.cameras[camera_name]
        positions, velocities, rotations = state.positions, state.velocities, state.rotations

        ray_dir = camera["ray_direction"]
        ray_offset = camera["ray_offset"]
        parent = camera["parent"]

        px_ver = ray_dir.shape[0]
        px_hor = ray_dir.shape[1]

        # WORKPOINT

        if parent:
            pid = self.objects[parent]
            # rotate and move the camera according to its parent
            ray_dir = theano_convert_model_to_world_coordinate_no_bias(ray_dir, rotations[:,pid,:,:])
            ray_offset = theano_convert_model_to_world_coordinate(ray_offset, rotations[:,pid,:,:], positions[:,pid,:])
        else:
            ray_dir = ray_dir[None,:,:,:]
            ray_offset = ray_offset[None,:,:,:]

        # step 2a: intersect the rays with all the spheres
        has_spheres = (0 != len(self.sphere_parent))
        #s_relevant = np.ones(shape=(self.batch_size, px_ver, px_hor, self.sphere_parent.shape[0]))
        if has_spheres:
            s_pos_vectors = positions[:,None,None,self.sphere_parent,:]
            s_rot_matrices = rotations[:,self.sphere_parent,:,:]

            L = s_pos_vectors - ray_offset[:,:,:,None,:]
            tca = T.sum(L * ray_dir[:,:,:,None,:],axis=4)  # L.dotProduct(ray_dir)
            #// if (tca < 0) return false;
            d2 = T.sum(L * L, axis=4) - tca*tca
            r2 = self.sphere_radius**2
            #if (d2 > radius2) return false;
            s_relevant = (tca > 0) * (d2[:,:,:,:] < r2[None,None,None,:])
            float_s_relevant = T.cast(s_relevant, 'float32')
            thc = T.sqrt((r2[None,None,None,:] - float_s_relevant * d2[:,:,:,:]))
            s_t0 = tca - thc
            Phit = ray_offset[:,:,:,None,:] + s_t0[:,:,:,:,None]*ray_dir[:,:,:,None,:]
            N = (Phit-s_pos_vectors) / self.sphere_radius[None,None,None,:,None]
            N = theano_convert_world_to_model_coordinate_no_bias(N, s_rot_matrices[:,None,None,:,:,:])

            # tex_y en tex_x in [-1,1]
            s_tex_x = T.arctan2(N[:,:,:,:,2], N[:,:,:,:,0])/np.pi
            s_tex_y = -1.+(2.-eps)*T.arccos(T.clip(N[:,:,:,:,1], -1.0, 1.0)) / np.pi



        # step 2b: intersect the rays with the cubes (cubes=planes)
        # step 2c: intersect the rays with the planes
        has_faces = (0 != len(self.face_parent))
        if has_faces:
            hasparent = [i for i,par in enumerate(self.face_parent) if par is not None]
            hasnoparent = [i for i,par in enumerate(self.face_parent) if par is None]
            parents = [parent for parent in self.face_parent if parent is not None]

            static_fn = numpy_repeat_new_axis(self.face_normal[hasnoparent,:], self.batch_size)
            static_fp = numpy_repeat_new_axis(self.face_point[hasnoparent,:], self.batch_size)
            static_ftx = numpy_repeat_new_axis(self.face_texture_x[hasnoparent,:], self.batch_size)
            static_fty = numpy_repeat_new_axis(self.face_texture_y[hasnoparent,:], self.batch_size)

            if hasparent:
                fn = theano_convert_model_to_world_coordinate_no_bias(self.face_normal[None,hasparent,:], rotations[:,parents,:,:])
                fn = T.concatenate([static_fn, fn], axis=1)
                fp = theano_convert_model_to_world_coordinate(self.face_point[None,hasparent,:], rotations[:,parents,:,:], positions[:,parents,:])
                fp = T.concatenate([static_fp, fp], axis=1)
                ftx = theano_convert_model_to_world_coordinate_no_bias(self.face_texture_x[None,hasparent,:], rotations[:,parents,:,:])
                ftx = T.concatenate([static_ftx, ftx], axis=1)
                fty = theano_convert_model_to_world_coordinate_no_bias(self.face_texture_y[None,hasparent,:], rotations[:,parents,:,:])
                fty = T.concatenate([static_fty, fty], axis=1)
            else:
                fn = static_fn
                fp = static_fp
                ftx = static_ftx
                fty = static_fty

            # reshuffle the face_texture_indexes to match the reshuffling we did above
            face_indices = hasnoparent + hasparent
            face_texture_index = self.face_texture_index[face_indices]
            face_texture_limited = self.face_texture_limited[face_indices]
            face_colors = self.face_colors[face_indices,:]

            denom = T.sum(fn[:,None,None,:,:] * ray_dir[:,:,:,None,:],axis=4)
            p0l0 = fp[:,None,None,:,:] - ray_offset[:,:,:,None,:]
            p_t0 = T.sum(p0l0 * fn[:,None,None,:,:], axis=4) / (denom + 1e-9)

            Phit = ray_offset[:,:,:,None,:] + p_t0[:,:,:,:,None]*ray_dir[:,:,:,None,:]

            pd = Phit-fp[:,None,None,:,:]
            p_tex_x = T.sum(ftx[:,None,None,:,:] * pd, axis=4)
            p_tex_y = T.sum(fty[:,None,None,:,:] * pd, axis=4)

            # the following only on limited textures
            p_relevant = (p_t0 > 0) * (1 - (1-(-1 < p_tex_x) * (p_tex_x < 1) * (-1 < p_tex_y) * (p_tex_y < 1)) * face_texture_limited)

            p_tex_x = ((p_tex_x+1)%2.)-1
            p_tex_y = ((p_tex_y+1)%2.)-1

        # step 3: find the closest point of intersection for all objects (z-culling)

        if has_spheres and has_faces:
            relevant = T.concatenate([s_relevant, p_relevant],axis=3).astype('float32')
            tex_x = T.concatenate([s_tex_x, p_tex_x],axis=3)
            tex_y = T.concatenate([s_tex_y, p_tex_y],axis=3)
            tex_t = np.concatenate([self.sphere_texture_index, face_texture_index],axis=0)
            t = T.concatenate([s_t0, p_t0], axis=3)
        elif has_spheres:
            relevant = s_relevant.astype('float32')
            tex_x = s_tex_x
            tex_y = s_tex_y
            tex_t = self.sphere_texture_index
            t = s_t0
        elif has_faces:
            relevant = p_relevant.astype('float32')
            tex_x = p_tex_x
            tex_y = p_tex_y
            tex_t = face_texture_index
            t = p_t0
        else:
            raise NotImplementedError()


        mint = T.min(t*relevant + (1.-relevant)*np.float32(1e9), axis=3)
        relevant *= (t<=mint[:,:,:,None])  #only use the closest object

        # step 4: go into the object's texture and get the corresponding value (see image transform)
        x_size, y_size = self.textures.shape[1] - 1, self.textures.shape[2] - 1

        tex_x = (tex_x + 1)*x_size/2.
        tex_y = (tex_y + 1)*y_size/2.
        x_idx = T.floor(tex_x)
        x_wgh = tex_x - x_idx
        y_idx = T.floor(tex_y)
        y_wgh = tex_y - y_idx

        # if the following are -2,147,483,648 or -9,223,372,036,854,775,808, you have NaN's
        x_idx, y_idx = T.cast(x_idx,'int64'), T.cast(y_idx,'int64')

        textures = T.TensorConstant(type=T.ftensor4, data=self.textures.astype('float32'), name='textures')

        sample= (   x_wgh  *    y_wgh )[:,:,:,:,None] * textures[tex_t[None,None,None,:],x_idx+1,y_idx+1,:] + \
                (   x_wgh  * (1-y_wgh))[:,:,:,:,None] * textures[tex_t[None,None,None,:],x_idx+1,y_idx  ,:] + \
                ((1-x_wgh) *    y_wgh )[:,:,:,:,None] * textures[tex_t[None,None,None,:],x_idx  ,y_idx+1,:] + \
                ((1-x_wgh) * (1-y_wgh))[:,:,:,:,None] * textures[tex_t[None,None,None,:],x_idx  ,y_idx  ,:]


        # multiply with color of object
        colors = np.concatenate([self.sphere_colors, face_colors],axis=0)
        if np.min(colors)!=1.:  # if the colors are actually used
            sample = colors[None,None,None,:,:] * sample

        # step 5: return this value

        image = T.sum(sample * relevant[:,:,:,:,None],axis=3)

        background_color = camera["background_color"]
        if background_color is not None:
            # find the rays for which no object was relevant. Make them background color
            background = background_color[None,None,None,:] * (1-T.max(relevant[:,:,:,:],axis=3))[:,:,:,None]
            image += background

        # do a dimshuffle to closer match the deep learning conventions
        image = image.dimshuffle(0,3,2,1)

        return image



    def evaluate(self, state, dt, motor_signals):

        positions, velocities, rotations = state.positions, state.velocities, state.rotations
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
        M00 = self.lower_inertia_inv
        M01 = T.zeros(shape=(self.batch_size,self.num_moving_bodies,3,3))
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
                r1 = parameters["joint_in_model1_coordinates"][None,:].astype('float32')
                r2 = parameters["joint_in_model2_coordinates"][None,:]

                r1x = theano_convert_model_to_world_coordinate_no_bias(r1, rotations[:,idx1,:,:])
                r2x = theano_convert_model_to_world_coordinate_no_bias(r2, rotations[:,idx2,:,:])
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

                ac = parameters['axis_in_model2_coordinates'][None,:]
                a = theano_convert_model_to_world_coordinate_no_bias(ac, rotations[:,idx2,:,:])

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)
                
                if follows_Newtons_third_law:
                    J[2*c_idx+0] = T.concatenate([-a, batched_zeros],axis=1)
                J[2*c_idx+1] = T.concatenate([ a, batched_zeros],axis=1)

                if follows_Newtons_third_law:
                    position = positions[:,idx2,:] - positions[:,idx1,:] - parameters['pos_init']
                else:
                    position = positions[:,idx2,:] - parameters['pos_init']

                motor_signal = motor_signals[:,parameters["motor_id"]]
                if "min" in parameters and "max" in parameters:
                    motor_min = (parameters["min"]/180. * np.pi)
                    motor_max = (parameters["max"]/180. * np.pi)
                    motor_signal = T.clip(motor_signal, motor_min, motor_max)

                error_signal = theano_dot_last_dimension_vectors(position - motor_signal, a)

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
                angle = parameters["angle"]/180. * np.pi

                ac = parameters['axis_in_model2_coordinates'][None,:]
                a = theano_convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], rotations[:,idx2,:,:])

                rot_current = theano_dot_last_dimension_matrices(rotations[:,idx2,:,:], rotations[:,idx1,:,:].dimshuffle(0,2,1))
                rot_init = parameters['rot_init'].T
                rot_diff = theano_dot_last_dimension_matrices(rot_current, rot_init)

                traces = rot_diff[:,0,0] + rot_diff[:,1,1] + rot_diff[:,2,2]

                theta2 = T.arccos(T.clip(0.5*(traces-1),-1+eps,1-eps))
                cross = rot_diff.dimshuffle(0,2,1) - rot_diff
                dot2 = cross[:,1,2] * ac[:,0] + cross[:,2,0] * ac[:,1] + cross[:,0,1] * ac[:,2]

                theta = ((dot2>0) * 2 - 1) * theta2

                batched_zeros = numpy_repeat_new_axis(np.zeros((3,), dtype='float32'), self.batch_size)

                if parameters["angle"] < 0:
                    if follows_Newtons_third_law:
                        J[2*c_idx+0] = T.concatenate([batched_zeros, -a])
                    J[2*c_idx+1] = T.concatenate([batched_zeros, a])
                else:
                    if follows_Newtons_third_law:
                        J[2*c_idx+0] = T.concatenate([batched_zeros, a])
                    J[2*c_idx+1] = T.concatenate([batched_zeros, -a])

                b_error[c_idx] = T.abs_(angle - theta)

                if parameters["angle"] > 0:
                    b_error[c_idx] = angle - theta
                    self.C[c_idx] = (theta > angle)
                else:
                    b_error[c_idx] = theta - angle
                    self.C[c_idx] = (theta < angle)

                c_idx += 1


            if constraint == "linear limit":

                offset = parameters["offset"]
                if follows_Newtons_third_law:
                    ac = parameters['axis_in_model1_coordinates'][None,:]
                    a = theano_convert_model_to_world_coordinate_no_bias(ac, rotations[:,idx1,:,:])
                else:
                    a = parameters['axis_in_model1_coordinates'][None,:]

                if offset < 0:
                    if follows_Newtons_third_law:
                        J[2*c_idx+0] = T.concatenate([-a, batched_zeros])
                    J[2*c_idx+1] = T.concatenate([a, batched_zeros])
                else:
                    if follows_Newtons_third_law:
                        J[2*c_idx+0] = T.concatenate([a, batched_zeros])
                    J[2*c_idx+1] = T.concatenate([-a, batched_zeros])

                if follows_Newtons_third_law:
                    position = positions[:,idx2,:] - positions[:,idx1,:] - parameters['pos_init']
                else:
                    position = positions[:,idx2,:] - parameters['pos_init']

                current_offset = theano_dot_last_dimension_vectors(position, a)

                if parameters["offset"] > 0:
                    b_error[c_idx] = offset - current_offset
                    self.C[c_idx] = (current_offset > offset)
                else:
                    b_error[c_idx] = offset - current_offset
                    self.C[c_idx] = (current_offset < offset)

                c_idx += 1

            if constraint == "ground":
                r = self.radii[idx1].astype('float32')
                J[2*c_idx+0] = numpy_repeat_new_axis(np.array([0,0,1,0,0,0], dtype='float32'), self.batch_size)
                J[2*c_idx+1] = numpy_repeat_new_axis(np.array([0,0,0,0,0,0], dtype='float32'), self.batch_size)

                # TODO: use T.maximum
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

        # TODO: batch-dot-product
        m_eff = 1./T.sum(T.sum(J[:,:,:,None,:]*mass_matrix, axis=4)*J, axis=(2,3))

        k = m_eff * (self.w**2)
        c = m_eff * 2*self.zeta*self.w

        CFM = 1./(c+dt*k)
        ERP = dt*k/(c+dt*k)
        m_c = 1./(1./m_eff + CFM)
        b = ERP/dt * b_error + b_res

        for iteration in xrange(self.projected_gauss_seidel_iterations):
            # this changes every iteration
            v = newv[:,zipped_indices,:].reshape(shape=(self.batch_size, self.num_constraints, 2, 6))

            lamb = - m_c * (T.sum(J*v, axis=(2,3)) + CFM * self.impulses_P + b)
            self.impulses_P += lamb

            if self.do_impulse_clipping:
                if self.do_friction_clipping:
                    clipping_force = self.impulses_P[:,self.clipping_idx]
                    clipping_limit = abs(self.clipping_a * clipping_force + self.clipping_b * dt)
                else:
                    clipping_limit = self.clipping_b * dt
                self.impulses_P = T.clip(self.impulses_P,-clipping_limit, clipping_limit)

            if self.has_conditional_constraints:
                applicable = (1.0*(C<=0)) * (1-(self.only_when_positive*(self.impulses_P<=0)))
                self.impulses_P = self.impulses_P * applicable

            # TODO: batch-dot-product
            result = T.sum(mass_matrix*J[:,:,:,None,:], axis=4) * self.impulses_P[:,:,None,None]
            result = result.reshape((self.batch_size, 2*self.num_constraints, 6))

            r = []
            for constraint in self.map_object_to_constraint:
                idx_list = np.array(constraint, dtype='int64')  # deal with empty lists
                delta_v = T.sum(result[:,idx_list,:], axis=1)
                r.append(delta_v)
            newv = newv + T.stack(r, axis=1)
        #print
        #print theano.printing.debugprint(newv)
        return newv

