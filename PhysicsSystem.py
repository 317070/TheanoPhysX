from collections import namedtuple
from math import pi
import json

__author__ = 'jonas'
import numpy as np
import scipy.linalg
import scipy.ndimage
import random
#np.seterr(all='raise')

X = 0
Y = 1
Z = 2

DTYPE = 'float32'

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


def convert_world_to_model_coordinate(coor, rot_matrix, model_position):
    return np.dot(rot_matrix, coor - model_position)

def convert_world_to_model_coordinate_no_bias(coor, rot_matrix):
    return np.dot(rot_matrix, coor)

def convert_model_to_world_coordinate(coor, rot_matrix, model_position):
    return convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position

def convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    return np.sum(rot_matrix[:,:] * coor[:,None], axis=0)


def batched_convert_world_to_model_coordinate(coor, rot_matrix, model_position):
    A = coor.ndim
    B = rot_matrix.ndim
    print A,B
    if A==4 and B==3:
        return batched_convert_world_to_model_coordinate_no_bias(coor[:,:,:,:] - model_position[None,None,:,:], rot_matrix)
    raise NotImplementedError()
    #return np.dot(rot_matrix, coor - model_position)

def batched_convert_world_to_model_coordinate_no_bias(coor, rot_matrix):
    A = coor.ndim
    B = rot_matrix.ndim
    print A,B
    if A==4 and B==3:
        return np.sum(rot_matrix[None,None,:,:,:] * coor[:,:,:,None,:], axis=4)
    raise NotImplementedError()
    #return np.dot(rot_matrix, coor)

def batched_convert_model_to_world_coordinate(coor, rot_matrix, model_position):
    A = coor.ndim
    B = rot_matrix.ndim
    print A,B
    if A==3 and B==2:
        return batched_convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position[None,None,:]
    if A==2 and B==2:
        return batched_convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position[None,:]
    if A==2 and B==3:
        return batched_convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position[:,:]

    raise NotImplementedError()
    #return batched_convert_model_to_world_coordinate_no_bias(coor, rot_matrix) + model_position

def batched_convert_model_to_world_coordinate_no_bias(coor, rot_matrix):
    A = coor.ndim
    B = rot_matrix.ndim
    print A,B
    if A==3 and B==2:
        return np.sum(rot_matrix[None,None,:,:] * coor[:,:,:,None], axis=2)
    if A==2 and B==2:
        return np.sum(rot_matrix[None,:,:] * coor[:,:,None], axis=1)
    if A==2 and B==3:
        return np.sum(rot_matrix[:,:,:] * coor[:,:,None], axis=1)
    raise NotImplementedError()
    #return np.sum(rot_matrix[:,:] * coor[:,None], axis=0)



def skew_symmetric(x):
    a,b,c = x[...,0,None,None],x[...,1,None,None],x[...,2,None,None]
    z = np.zeros(x.shape[:-1]+(1,1))
    return np.concatenate([
                    np.concatenate(( z,-c, b),axis=-1),
                    np.concatenate(( c, z,-a),axis=-1),
                    np.concatenate((-b, a, z),axis=-1)
                            ],axis=-2)


class Rigid3DBodyEngine(object):
    def __init__(self):
        self.radii = np.zeros(shape=(0,), dtype=DTYPE)
        self.dimensions = np.zeros(shape=(0,3), dtype=DTYPE)
        self.positionVectors = np.zeros(shape=(0,3), dtype=DTYPE)
        self.rot_matrices = np.zeros(shape=(0,3,3), dtype='float32')
        self.velocityVectors = np.zeros(shape=(0,6), dtype=DTYPE)
        self.massMatrices = np.zeros(shape=(0,6,6), dtype=DTYPE)
        self.objects = {None:None}
        self.shapes = []
        self.constraints = []
        self.sensors = []

        self.planes = dict()
        self.planeNormals = np.zeros(shape=(0,3), dtype=DTYPE)
        self.planePoints = np.zeros(shape=(0,3), dtype=DTYPE)

        self.P = None
        self.w = None
        self.zeta = None
        self.only_when_positive = None
        self.map_object_to_constraint = None
        self.clipping_a   = None
        self.clipping_idx = None
        self.clipping_b   = None
        self.C = None
        self.zero_index = None
        self.one_index = None

        self.DT = None
        self.projected_gauss_seidel_iterations = None
        self.rotation_reorthogonalization_iterations = None
        self.warm_start = None

        self.face_normal = np.zeros(shape=(0,3), dtype=DTYPE)
        self.face_point = np.zeros(shape=(0,3), dtype=DTYPE)
        self.face_parent = []
        self.face_texture_x = np.zeros(shape=(0,3), dtype=DTYPE)
        self.face_texture_y = np.zeros(shape=(0,3), dtype=DTYPE)
        self.face_texture_limited = np.zeros(shape=(0,), dtype='int32')
        self.face_texture_index = np.zeros(shape=(0,), dtype='int32')
        self.face_colors = np.zeros(shape=(0,3), dtype=DTYPE)

        self.sphere_radius = np.zeros(shape=(0,), dtype=DTYPE)
        self.sphere_parent = np.zeros(shape=(0,), dtype='int32')
        self.sphere_texture_index = np.zeros(shape=(0,), dtype='int32')
        self.sphere_colors = np.zeros(shape=(0,3), dtype=DTYPE)

        self.textures = None
        self.texture_files = []

        self.cameras = dict()

    def set_integration_parameters(self,
                                   time_step=0.001,
                                   projected_gauss_seidel_iterations=1,
                                   rotation_reorthogonalization_iterations=1,
                                   warm_start=0,
                                   universe=False):
        self.DT = time_step
        self.projected_gauss_seidel_iterations = projected_gauss_seidel_iterations
        self.rotation_reorthogonalization_iterations = rotation_reorthogonalization_iterations
        self.warm_start = warm_start
        if universe:
            self.add_universe()

    # TODO: remove universe, replace with None in joints
    def add_universe(self, **parameters):
        self.objects["universe"] = self.positionVectors.shape[0]
        self.radii = np.append(self.radii, -1)
        self.dimensions = np.append(self.dimensions, [[-1,-1,-1]], axis=0)

        self.massMatrices = np.append(self.massMatrices, 1e6*np.diag([1,1,1,0.4,0.4,0.4])[None,:,:], axis=0)
        self.positionVectors = np.append(self.positionVectors, np.array([[0,0,0,1,0,0,0]], dtype=DTYPE), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([[0,0,0,0,0,0]], dtype=DTYPE), axis=0)
        self.addConstraint("universe", ["universe", "universe"], parameters={"f":1, "zeta":0.01})


    def addCube(self,
                reference,
                dimensions=[1,1,1],
                mass_density=1000.0,
                position=[0,0,0],
                rotation=[1,0,0,0],
                velocity=[0,0,0,0,0,0],
                visible=True,
                default_faces={
                    "texture": None,
                    "color": [1,1,1],
                    "visible": True
                                 },
                faces = {},
                **kwargs):
        self.objects[reference] = self.positionVectors.shape[0]
        self.shapes += ["cube"]
        self.radii = np.append(self.radii, -1)
        self.dimensions = np.append(self.dimensions, [dimensions], axis=0)

        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype=DTYPE), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype=DTYPE), axis=0)

        mass = mass_density*np.prod(dimensions)
        I1 = 1./12. * (dimensions[1]**2 + dimensions[2]**2)
        I2 = 1./12. * (dimensions[0]**2 + dimensions[2]**2)
        I3 = 1./12. * (dimensions[0]**2 + dimensions[1]**2)

        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,I1,I2,I3])[None,:,:], axis=0)

        if visible:
            for i,tag in enumerate(["right", "left", "front", "back", "top", "bottom"]):
                dt = dict(default_faces)  # take a copy, don't mute original
                if tag in faces:
                    face = faces[tag]
                    if face is None:
                        continue
                    dt.update(face)
                if "visible" in dt and not dt["visible"]:
                    continue
                # This is all in model coordinates
                nn = [[1, 0, 0],
                     [-1, 0, 0],
                      [0, 1, 0],
                      [0,-1, 0],
                      [0, 0, 1],
                      [0, 0, -1]][i]

                h, w, z = dimensions
                h, w, z = h/2., w/2., z/2.
                np0 = [[h, 0, 0],
                       [-h, 0, 0],
                       [0, w, 0],
                       [0, -w, 0],
                       [0, 0, z],
                       [0, 0, -z]][i]

                nx0 = [[0, w, 0],
                     [0,-w, 0],
                     [0, 0, z],
                     [0, 0,-z],
                     [h, 0, 0],
                     [-h, 0,0]][i]
                ny0 = [[0, 0, z],
                     [0, 0,-z],
                     [h, 0, 0],
                     [-h,0, 0],
                     [0, w, 0],
                     [0,-w, 0]][i]

                texture_file = dt["texture"]
                color = dt["color"] if "color" in dt else [1,1,1]
                self.addFace(normal=nn, point=np0, parent=reference, face_x=nx0, face_y=ny0, texture=texture_file, color=color, limited=True)


    def addSphere(self,
                  reference,
                  radius=1.0,
                  mass_density=1000.0,
                  position=[0,0,0],
                  rotation=[1,0,0,0],
                  velocity=[0,0,0,0,0,0],
                  visible=True,
                  texture=None,
                  color=[1,1,1],
                  **kwargs):
        self.objects[reference] = self.positionVectors.shape[0]
        self.shapes += ["sphere"]
        self.radii = np.append(self.radii, radius)
        self.dimensions = np.append(self.dimensions, [[-1,-1,-1]], axis=0)
        self.positionVectors = np.append(self.positionVectors, np.array([position], dtype=DTYPE), axis=0)
        self.rot_matrices = np.append(self.rot_matrices, np.array([quat_to_rot_matrix(rotation)], dtype='float32'), axis=0)
        self.velocityVectors = np.append(self.velocityVectors, np.array([velocity], dtype=DTYPE), axis=0)
        mass = mass_density*4./3.*np.pi*radius**3
        self.massMatrices = np.append(self.massMatrices, mass*np.diag([1,1,1,0.4,0.4,0.4])[None,:,:], axis=0)

        if visible:
            self.sphere_radius = np.append(self.sphere_radius, [radius], axis=0)
            self.sphere_parent = np.append(self.sphere_parent, [self.objects[reference]], axis=0)
            texture_index = self.load_texture(texture)
            self.sphere_texture_index = np.append(self.sphere_texture_index, [texture_index], axis=0)
            self.sphere_colors = np.append(self.sphere_colors, [color], axis=0)


    def addPlane(self, reference, normal, position, visible=True, **kwargs):
        self.objects[reference] = None
        self.planes[reference] = reference
        self.planeNormals = np.append(self.planeNormals, np.array([normal]), axis=0)
        self.planePoints = np.append(self.planePoints, np.array([position]), axis=0)

        if visible:
            self.addFace(normal=normal, point=position, parent=None, **kwargs)


    def addFace(self,
                normal=[0,0,1],
                point=[0,0,0],
                parent=None,
                face_x=None,
                face_y=None,
                limited=False,
                texture=None,
                color=[1,1,1],
                **kwargs):

        if face_x is None or face_y is None:
            if (normal == np.array([1,0,0])).all():
                face_x = np.array([0,1,0], dtype=DTYPE)
                face_y = np.array([0,0,1], dtype=DTYPE)
            else:
                face_x = np.array([0,-normal[2],normal[1]])
                face_y = np.cross(normal, face_x)
        else:
            face_x, face_y = np.array(face_x, dtype=DTYPE), np.array(face_y, dtype=DTYPE)
            print face_x,face_y
            face_x = np.divide(np.ones_like(face_x), face_x, out=np.zeros_like(face_x), where=(face_x!=0))
            face_y = np.divide(np.ones_like(face_y), face_y, out=np.zeros_like(face_y), where=(face_y!=0))
            print face_x,face_y

        self.face_normal = np.append(self.face_normal, [normal], axis=0)
        self.face_point = np.append(self.face_point, [point], axis=0)
        self.face_parent.append(self.objects[parent] if parent is not None else parent)
        self.face_texture_x = np.append(self.face_texture_x, [face_x], axis=0)
        self.face_texture_y = np.append(self.face_texture_y, [face_y], axis=0)
        self.face_texture_limited = np.append(self.face_texture_limited, [1 if limited else 0], axis=0)

        texture_index = self.load_texture(texture)
        self.face_texture_index = np.append(self.face_texture_index, [texture_index], axis=0)
        self.face_colors = np.append(self.face_colors, [color], axis=0)


    def load_texture(self, filename):

        if filename in self.texture_files:
            return self.texture_files.index(filename)

        if filename is None:
            if self.textures is None:
                self.textures = np.ones(shape=(1,128,128,3))
            else:
                self.textures = np.append(self.textures, [np.zeros(shape=self.textures.shape[1:])], axis=0)
        else:
            tex = scipy.ndimage.imread(filename).transpose(1,0,2)[:,::-1,:3] / 255.

            assert tex.ndim==3 and tex.shape[2]==3, "Make sure the texture is in RGB-space"
            assert (0.0<=tex).all() and (tex<=1.0).all()
            if self.textures is None:
                self.textures = tex[None,:,:,:]
            elif np.min(self.textures)==1 and self.textures.shape[0]==1:
                # we already added zeros and nothing else, reshape the texture to fit this textures shape
                zeros = np.zeros(shape=tex.shape)
                self.textures = np.append([zeros], [tex], axis=0)
            else:
                self.textures = np.append(self.textures, [tex], axis=0)

        self.texture_files.append(filename)
        return self.textures.shape[0]-1


    def addCamera(self,
                  reference,
                  parent=None,
                  position=[0,0,0],  # position of the lens
                  orientation=[1,0,0,0],  # orientation of the camera, by default, camera looks according to the x-axis
                  focal_length=0.043,
                  width=0.035,  # width of the sensor in meters
                  height=0.035,  # height of the sensor in meters
                  horizontal_pixels=128,
                  vertical_pixels=128,
                  background_color=[0.0, 191.0/255.0, 1.0],  # color of the sky
                  ):

        camera = dict()
        self.cameras[reference] = camera
        position = np.array(position)
        if parent is not None:
            parent_id = self.objects[parent]
            camera_position = position - self.positionVectors[parent_id,:] # in model coordinates
        else:
            camera_position = position

        camera["parent"] = parent

        dcw = width/(horizontal_pixels*2)
        dch = height/(vertical_pixels*2)

        ray_dir = np.array(np.meshgrid([focal_length],
                                       np.linspace(-0.5*width+dcw, 0.5*width-dcw, horizontal_pixels),
                                       np.linspace(-0.5*height+dch,0.5*height-dch,vertical_pixels),
                                       ))[:,:,0,:].T

        # add the inital orientation
        ray_dir = batched_convert_model_to_world_coordinate_no_bias(ray_dir, quat_to_rot_matrix(orientation))

        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=2, keepdims=True)   #normalize
        if focal_length<0: #support negative focal lengths
            ray_dir = -ray_dir

        # the following is useful for depth of field !
        """
        ray_offset = np.array(np.meshgrid([0],
                                       0*np.linspace(-0.5*camera_width+dcw, 0.5*camera_width-dcw, horizontal_pixels),
                                       0*np.linspace(-0.5*camera_height+dch,0.5*camera_height-dch,vertical_pixels),
                                       ))[:,:,0,:].T
        """
        ray_offset = np.zeros(shape=(3,))[None,None,:] #"""
        ray_offset = batched_convert_model_to_world_coordinate_no_bias(ray_offset, quat_to_rot_matrix(orientation))
        ray_offset = ray_offset + camera_position[None,None,:]

        camera["ray_offset"] = ray_offset
        camera["ray_direction"] = ray_dir
        camera["background_color"] = np.array(background_color)




    def getCameraImage(self, camera_name):

        # TODO: flatten the for loop
        # TODO: move a lot of the initialization outside of this function
        # TODO: edit the json-format to support all of this
        # TODO: support multiple cameras
        # TODO: support the visible-flag
        # TODO: support the physics-flag
        # TODO: support initial camera rotation relative to model

        # get camera image
        # do ray-sphere and ray-plane intersections
        # find cube by using 6 planes, throwing away the irrelevant intersections

        # step 1: generate list of rays (1 per pixel)
        # focal_point (3,)
        # ray_dir (px_hor, px_ver, 3)
        # ray_offset (px_hor, px_ver, 3)

        camera = self.cameras[camera_name]


        ray_dir = camera["ray_direction"]
        ray_offset = camera["ray_offset"]
        parent = camera["parent"]

        px_ver = ray_dir.shape[0]
        px_hor = ray_dir.shape[1]
        if parent:
            pid = self.objects[parent]
            # rotate and move the camera according to its parent
            ray_dir = batched_convert_model_to_world_coordinate_no_bias(ray_dir, self.rot_matrices[pid,:,:])
            ray_offset = batched_convert_model_to_world_coordinate(ray_offset, self.rot_matrices[pid,:,:], self.positionVectors[pid,:])

        # step 2a: intersect the rays with all the spheres
        s_relevant = np.ones(shape=(px_ver, px_hor, self.sphere_parent.shape[0]))
        s_pos_vectors = self.positionVectors[None,None,self.sphere_parent,:]
        s_rot_matrices = self.rot_matrices[self.sphere_parent,:,:]

        L = s_pos_vectors - ray_offset[:,:,None,:]
        tca = np.sum(L * ray_dir[:,:,None,:],axis=3)  # L.dotProduct(ray_dir);
        #// if (tca < 0) return false;
        s_relevant *= (tca > 0)
        d2 = np.sum(L * L, axis=3) - tca*tca
        r2 = self.sphere_radius**2
        #if (d2 > radius2) return false;

        s_relevant *= (d2 <= r2)

        thc = np.sqrt(s_relevant * (r2 - d2))
        s_t0 = tca - thc
        Phit = ray_offset[:,:,None,:] + s_t0[:,:,:,None]*ray_dir[:,:,None,:]
        N = (Phit-s_pos_vectors) / self.sphere_radius[None,None,:,None]

        N = batched_convert_world_to_model_coordinate_no_bias(N, s_rot_matrices)

        s_tex_x = np.arctan2(N[:,:,:,2], N[:,:,:,0])/np.pi
        s_tex_y = -1+2*np.arccos(N[:,:,:,1]*s_relevant)/np.pi
        # tex_y en tex_x in [-1,1]


        # step 2b: intersect the rays with the cubes

        # step 2c: intersect the rays with the planes
        p_relevant = np.ones(shape=(px_ver, px_hor, self.face_normal.shape[0]))

        fn = np.array(self.face_normal[:,:])
        fp = np.array(self.face_point[:,:])
        ftx = np.array(self.face_texture_x[:,:])
        fty = np.array(self.face_texture_y[:,:])
        hasparent = [i for i,par in enumerate(self.face_parent) if par is not None]
        parents = [parent for parent in self.face_parent if parent is not None]
        print parents
        fn[hasparent,:] = batched_convert_model_to_world_coordinate_no_bias(fn[hasparent,:], self.rot_matrices[parents,:,:])
        fp[hasparent,:] = batched_convert_model_to_world_coordinate(fp[hasparent,:], self.rot_matrices[parents,:,:], self.positionVectors[parents,:])
        ftx[hasparent,:] = batched_convert_model_to_world_coordinate_no_bias(ftx[hasparent,:], self.rot_matrices[parents,:,:])
        fty[hasparent,:] = batched_convert_model_to_world_coordinate_no_bias(fty[hasparent,:], self.rot_matrices[parents,:,:])


        denom = np.sum(fn[None,None,:,:] * ray_dir[:,:,None,:],axis=3)
        p0l0 = fp[None,None,:,:] - ray_offset[:,:,None,:]
        p_t0 = np.sum(p0l0 * fn[None,None,:,:], axis=3) / (denom + 1e-9)
        p_relevant *= (p_t0 > 0)  #only planes in front of us

        Phit = ray_offset[:,:,None,:] + p_t0[:,:,:,None]*ray_dir[:,:,None,:]

        pd = Phit-fp
        p_tex_x = np.sum(ftx[None,None,:,:] * pd, axis=3)
        p_tex_y = np.sum(fty[None,None,:,:] * pd, axis=3)

        # the following only on limited textures
        p_relevant *= 1 - (1-(-1 < p_tex_x) * (p_tex_x < 1) * (-1 < p_tex_y) * (p_tex_y < 1)) * self.face_texture_limited

        p_tex_x = ((p_tex_x+1)%2.)-1
        p_tex_y = ((p_tex_y+1)%2.)-1

        # step 3: find the closest point of intersection (z-culling) for all objects
        relevant = np.concatenate([s_relevant, p_relevant],axis=2)
        tex_x = np.concatenate([s_tex_x, p_tex_x],axis=2)
        tex_y = np.concatenate([s_tex_y, p_tex_y],axis=2)
        tex_t = np.concatenate([self.sphere_texture_index, self.face_texture_index],axis=0)

        t = np.concatenate([s_t0, p_t0], axis=2)

        mint = np.min(t*relevant + (1-relevant)*1e9, axis=-1)
        relevant *= (t==mint[:,:,None])  #only use the closest object

        # step 4: go into the object's texture and get the corresponding value (see image transform)
        x_size, y_size = self.textures.shape[1] - 1, self.textures.shape[2] - 1

        tex_x = (tex_x + 1)*x_size/2.
        tex_y = (tex_y + 1)*y_size/2.
        x_idx = np.floor(tex_x).astype('int32')
        x_wgh = tex_x - x_idx
        y_idx = np.floor(tex_y).astype('int32')
        y_wgh = tex_y - y_idx

        #print np.min(tex_x), np.max(tex_x)
        #print np.min(x_idx), np.max(x_idx)
        print np.min(self.textures), np.max(self.textures)

        sample= (   x_wgh  *    y_wgh )[:,:,:,None] * self.textures[tex_t[None,None,:],x_idx+1,y_idx+1,:] + \
                (   x_wgh  * (1-y_wgh))[:,:,:,None] * self.textures[tex_t[None,None,:],x_idx+1,y_idx  ,:] + \
                ((1-x_wgh) *    y_wgh )[:,:,:,None] * self.textures[tex_t[None,None,:],x_idx  ,y_idx+1,:] + \
                ((1-x_wgh) * (1-y_wgh))[:,:,:,None] * self.textures[tex_t[None,None,:],x_idx  ,y_idx  ,:]


        #print sample.shape
        # multiply with color of object
        colors = np.concatenate([self.sphere_colors, self.face_colors],axis=0)
        if np.min(colors)!=1:  # if the colors are actually used
            sample = colors[None,None,:,:] * sample

        print colors

        # step 5: return this value

        image = np.sum(sample * relevant[:,:,:,None],axis=2)

        background_color = camera["background_color"]
        if background_color is not None:
            # find the rays for which no object was relevant. Make them background color
            background = background_color[None,None,:] * (1-np.max(relevant[:,:,:],axis=2))[:,:,None]
            image += background

        return image



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

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx1,:,:], self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx2,:,:], self.positionVectors[idx2,:])

        self.addConstraint("ball-and-socket", [object1, object2], parameters)


    def addHingeConstraint(self, jointname, object1, object2, point, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx1,:,:], self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx2,:,:], self.positionVectors[idx2,:])

        # create two forbidden axis:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        if (axis == np.array([1,0,0])).all():
            forbidden_axis_1 = np.array([0,1,0], dtype=DTYPE)
            forbidden_axis_2 = np.array([0,0,1], dtype=DTYPE)
        else:
            forbidden_axis_1 = np.array([0,-axis[2],axis[1]])
            forbidden_axis_2 = np.cross(axis, forbidden_axis_1)

        parameters['axis'] = axis
        parameters['axis1_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_1, self.rot_matrices[idx1,:,:])
        parameters['axis2_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(forbidden_axis_2, self.rot_matrices[idx1,:,:])
        parameters['axis_in_model2_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx2,:,:])

        self.addConstraint("hinge", [object1, object2], parameters)



    def addSliderConstraint(self, jointname, object1, object2, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['q_init'] = q_div(self.positionVectors[idx2,3:], self.positionVectors[idx1,3:])

        self.addConstraint("slider", [object1, object2], parameters)

    def addFixedConstraint(self, jointname, object1, object2, point, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        parameters['joint_in_model1_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx1,:,:], self.positionVectors[idx1,:])
        parameters['joint_in_model2_coordinates'] = convert_world_to_model_coordinate(point, self.rot_matrices[idx2,:,:], self.positionVectors[idx2,:])

        self.addConstraint("fixed", [object1, object2], parameters)


    def addMotorConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2] if object2 else None

        # create two forbidden axis:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis
        parameters['axis_in_model2_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx2,:,:])

        self.addConstraint("motor", [object1, object2], parameters)


    def addLimitConstraint(self, object1, object2, axis, **parameters):
        idx1 = self.objects[object1]
        idx2 = self.objects[object2]

        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        parameters['axis'] = axis
        parameters['axis_in_model1_coordinates'] = convert_world_to_model_coordinate_no_bias(axis, self.rot_matrices[idx1,:,:])

        self.addConstraint("limit", [object1, object2], parameters)


    def addSensor(self, **kwargs):
        if "axis" in kwargs and "reference" in kwargs:
            kwargs["axis"] = convert_world_to_model_coordinate_no_bias(kwargs["axis"], self.rot_matrices[self.getObjectIndex(kwargs["reference"]),:,:])
        kwargs["axis"] = np.array(kwargs["axis"], dtype='float32')
        self.sensors.append(kwargs)


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
            elif primitive["shape"] == "plane":
                self.addPlane(elementname, **parameters)
            elif primitive["shape"] == "face":
                self.addFace(**parameters)

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

        for cameraname, camera in robot_dict["cameras"].iteritems():
            primitive = element[0]
            parameters = dict(robot_dict["default_camera_parameters"])  # copy
            parameters.update(camera)
            self.addCamera(cameraname, **camera)

        for sensor in robot_dict["sensors"]:
            self.addSensor(**sensor)


    def compile(self):

        self.inertia_inv = np.linalg.inv(self.massMatrices)
        self.num_constraints = 0
        for (constraint,references,parameters) in self.constraints:
            if constraint == "universe":
                self.num_constraints += 6
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
            if constraint == "ground" and parameters["torsional_friction"] and parameters["mu"]!=0:
                self.num_constraints += 1
            if constraint == "limit":
                self.num_constraints += 1
            if constraint == "motor":
                self.num_constraints += 1

        self.P = np.zeros((self.num_constraints,), dtype=DTYPE)# constant
        self.w = np.zeros((self.num_constraints,), dtype=DTYPE)  # 0 constraints
        self.zeta = np.zeros((self.num_constraints,), dtype=DTYPE)  # 0 constraints

        self.only_when_positive = np.zeros((self.num_constraints,), dtype=DTYPE)  # 0 constraints
        self.map_object_to_constraint = [[] for _ in xrange(len(self.objects))]
        self.clipping_a   = np.ones((self.num_constraints,), dtype=DTYPE)
        self.clipping_idx = range(self.num_constraints)
        self.clipping_b   = np.zeros((self.num_constraints,), dtype=DTYPE)

        self.zero_index = []
        self.one_index = []
        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            idx2 = references[1]
            if constraint == "universe":
                for i in xrange(6):
                    self.map_object_to_constraint[idx1].append(2*(c_idx+i) + 0)
                    self.zero_index.append(idx1)
                    self.one_index.append(idx2)

                self.w[c_idx:c_idx+6] = parameters["f"] * 2*np.pi
                self.zeta[c_idx:c_idx+6] = parameters["zeta"]
                c_idx += 6

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



    def evaluate(self, dt, positions, velocities, motor_signals):

        # ALL CONSTRAINTS CAN BE TRANSFORMED TO VELOCITY CONSTRAINTS!
        ##################
        # --- Step 1 --- #
        ##################
        # First, we integrate the applied force F_a acting of each rigid body (like gravity, ...) and
        # we obtain some new velocities v2' that tends to violate the constraints.

        totalforce = np.array([0,0,0,0,0,0], dtype=DTYPE)  # total force acting on body outside of constraints
        acceleration = np.array([0,0,-9.81,0,0,0], dtype=DTYPE)  # acceleration of the default frame
        newv = velocities + dt * (np.dot(self.massMatrices, totalforce) + acceleration[None,:])
        originalv = newv.copy()

        interesting = []

        ##################
        # --- Step 2 --- #
        ##################
        # now enforce the constraints by having corrective impulses
        # convert mass matrices to world coordinates
        M = np.zeros_like(self.inertia_inv, dtype=DTYPE)

        M[:,:3,:3] = self.inertia_inv[:,:3,:3]
        M[:,3:,3:] = np.sum(self.rot_matrices[:,:,None,:,None] * self.inertia_inv[:,3:,3:,None,None] * self.rot_matrices[:,None,:,None,:], axis=(1,2))

        #"""
        # changes every timestep
        J = np.zeros((self.num_constraints,2,6), dtype=DTYPE)  # 0 constraints x 2 objects x 6 states
        b_res = np.zeros((self.num_constraints,), dtype=DTYPE)  # 0 constraints
        b_error = np.zeros((self.num_constraints,), dtype=DTYPE)  # 0 constraints
        self.C = np.ones((self.num_constraints,), dtype=DTYPE)  # 0 constraints

        c_idx = 0
        for constraint,references,parameters in self.constraints:
            idx1 = references[0]
            if references[1] is not None:
                idx2 = references[1]
            else:
                idx2 = None

            if constraint == "universe":
                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([-np.eye(3), np.zeros((3,3))])[:,i]
                    #J[c_idx+i,1,:] = np.concatenate([ np.eye(3), np.zeros((3,3))])[:,i]
                b_error[c_idx:c_idx+3] = -positions[idx1,:3]
                c_idx += 3

                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE),-np.eye(3)])[:,i]
                    #J[c_idx+i,1,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE), np.eye(3)])[:,i]

                rot_diff =  self.rot_matrices[idx1,:,:]
                cross = rot_diff - rot_diff.T

                b_error[c_idx] = 0#cross[1,2]
                b_error[c_idx+1] = 0#cross[2,0]
                b_error[c_idx+2] = 0#cross[0,1]

                c_idx += 3

            if constraint == "ball-and-socket" or constraint == "hinge" or constraint == "fixed":
                r1x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model1_coordinates"], self.rot_matrices[idx1,:,:])
                r2x = convert_model_to_world_coordinate_no_bias(parameters["joint_in_model2_coordinates"], self.rot_matrices[idx2,:,:])
                ss_r1x = skew_symmetric(r1x)
                ss_r2x = skew_symmetric(r2x)
                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([-np.eye(3), ss_r1x])[:,i]
                    J[c_idx+i,1,:] = np.concatenate([ np.eye(3),-ss_r2x])[:,i]

                b_error[c_idx:c_idx+3] = (positions[idx2,:3]+r2x-positions[idx1,:3]-r1x)
                c_idx += 3

            if constraint == "slider" or constraint == "fixed":

                # something is badly interacting here!

                for i in xrange(3):
                    J[c_idx+i,0,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE),-np.eye(3)])[:,i]
                    J[c_idx+i,1,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE), np.eye(3)])[:,i]
                    #J[c_idx+i,0,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE),-self.rot_matrices[idx1,:,:].T])[:,i]
                    #J[c_idx+i,1,:] = np.concatenate([np.zeros((3,3), dtype=DTYPE), self.rot_matrices[idx2,:,:].T])[:,i]


                rot_current = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, parameters['rot_init'].T)
                cross = rot_diff.T - rot_diff

                # TODO: find b_error for rotational matrices!
                #q_current = q_div(positions[idx2,3:], positions[idx1,3:])
                #q_diff = q_div(q_current, parameters['q_init'])
                #b_error[c_idx:c_idx+3] = 2*q_diff[1:]
                # TODO: THIS ACTUALLY WORKS! (NOT STABLE ENOUGH)
                b_error[c_idx] = 0#cross[1,2]
                b_error[c_idx+1] = 0#cross[2,0]
                b_error[c_idx+2] = 0#cross[0,1]
                #print b_error[c_idx:c_idx+3]
                c_idx += 3

            if constraint == "hinge":

                a2x = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], self.rot_matrices[idx2,:,:])
                b1x = convert_model_to_world_coordinate_no_bias(parameters['axis1_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                c1x = convert_model_to_world_coordinate_no_bias(parameters['axis2_in_model1_coordinates'], self.rot_matrices[idx1,:,:])

                ss_a2x = skew_symmetric(a2x)

                if idx1==2:
                    """print "axis:", a2x
                    print "forbidden:", b1x
                    print "forbidden:", c1x
                    print np.dot(a2x, b1x), np.dot(a2x, c1x), np.dot(b1x, c1x)
                    print ss_a2x"""

                J[c_idx+0,0,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),-np.dot(b1x,ss_a2x)])
                J[c_idx+0,1,:] = np.concatenate([np.zeros((3,), dtype=DTYPE), np.dot(b1x,ss_a2x)])
                J[c_idx+1,0,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),-np.dot(c1x,ss_a2x)])
                J[c_idx+1,1,:] = np.concatenate([np.zeros((3,), dtype=DTYPE), np.dot(c1x,ss_a2x)])

                b_error[c_idx:c_idx+2] = np.array([np.sum(a2x*b1x),np.sum(a2x*c1x)])
                c_idx += 2

            if constraint == "limit":
                angle = parameters["angle"]/180. * np.pi
                a = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model1_coordinates'], self.rot_matrices[idx1,:,:])
                rot_current = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, quat_to_rot_matrix(parameters['q_init']).T)
                theta2 = np.arccos(0.5*(np.trace(rot_diff)-1))
                cross = rot_diff.T - rot_diff
                dot2 = cross[1,2] * a[0] + cross[2,0] * a[1] + cross[0,1] * a[2]
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

            if constraint == "motor":

                interesting.append(c_idx)
                ac = parameters['axis_in_model2_coordinates']
                a = convert_model_to_world_coordinate_no_bias(parameters['axis_in_model2_coordinates'], self.rot_matrices[idx2,:,:])

                rot_current = np.dot(self.rot_matrices[idx2,:,:], self.rot_matrices[idx1,:,:].T)
                rot_diff = np.dot(rot_current, parameters['rot_init'].T)
                theta2 = np.arccos(np.clip(0.5*(np.trace(rot_diff)-1),-1,1))
                cross = rot_diff.T - rot_diff
                dot2 = cross[1,2] * ac[0] + cross[2,0] * ac[1] + cross[0,1] * ac[2]

                theta = theta2 * ((dot2>0) * 2 - 1)

                J[c_idx,0,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),-a])
                J[c_idx,1,:] = np.concatenate([np.zeros((3,), dtype=DTYPE),a])

                motor_signal = motor_signals[parameters["motor_id"]]

                if "min" in parameters and "max" in parameters:
                    motor_signal = np.clip(motor_signal, parameters["min"]/180. * np.pi, parameters["max"]/180. * np.pi)

                def smallestSignedAngleBetween(x, y):
                    a1 = (x - y) % (2*np.pi)
                    b1 = (y - x) % (2*np.pi)
                    return min(a1,b1)*((a1>b1)*2-1)

                error_signal = -smallestSignedAngleBetween(theta, motor_signal)

                if parameters["type"] == "velocity":
                    b_error[c_idx] = motor_signal
                elif parameters["type"] == "position":
                    if "delta" in parameters and "motor_velocity" in parameters:
                        velocity = parameters["motor_velocity"] / 180. * np.pi
                        b_error[c_idx] = dt * np.clip((abs(error_signal) > parameters["delta"]) * error_signal * parameters["motor_gain"], -velocity, velocity)
                    else:
                        b_error[c_idx] = dt * error_signal * parameters["motor_gain"]

                #print "%.3f\t%.3f\t%.3f\t%.3f" %(theta, theta2, b_error[c_idx], motor_signal)
                #print c_idx

                c_idx += 1

            if constraint == "ground":
                r = self.radii[idx1]
                J[c_idx,0,:] = np.array([0,0,1,0,0,0], dtype=DTYPE)
                J[c_idx,1,:] = np.array([0,0,0,0,0,0], dtype=DTYPE)

                b_error[c_idx] = np.min([positions[idx1,Z] - r + parameters["delta"],0])
                b_res[c_idx] = parameters["alpha"] * newv[idx1,Z]
                self.C[c_idx] = (positions[idx1,Z] - r < 0.0)
                c_idx += 1

            if constraint == "ground" and parameters["mu"]!=0:
                r = self.radii[idx1]
                for i in xrange(2):
                    if i==0:
                        J[c_idx+i,0,:] = np.array([0,1,0,-r,0,0], dtype=DTYPE)
                    else:
                        J[c_idx+i,0,:] = np.array([1,0,0,0,r,0], dtype=DTYPE)
                    J[c_idx+i,1,:] = np.array([0,0,0,0,0,0], dtype=DTYPE)
                    self.C[c_idx+i] = (positions[idx1,Z] - r < 0.0)
                c_idx += 2

            if constraint == "ground" and parameters["torsional_friction"] and parameters["mu"]!=0:
                r = self.radii[idx1]
                J[c_idx,0,:] = np.array([0,0,0,0,0,r], dtype=DTYPE)
                J[c_idx,1,:] = np.array([0,0,0,0,0,0], dtype=DTYPE)
                self.C[c_idx] = (positions[idx1,Z] - r < 0.0)
                b_res[c_idx] = 0
                c_idx += 1
        print
        mass_matrix = np.concatenate((M[self.zero_index,None,:,:], M[self.one_index,None,:,:]), axis=1)

        #v = np.zeros((self.num_constraints,2,6))  # 0 constraints x 2 objects x 6 states
        #self.P = np.zeros((self.num_constraints,))# constant
        # warm start
        self.P = self.warm_start * self.P


        # http://myselph.de/gamePhysics/equalityConstraints.html
        for iteration in xrange(self.projected_gauss_seidel_iterations):
            v = np.concatenate((newv[self.zero_index,None,:],newv[self.one_index,None,:]),axis=1)

            m_eff = 1./np.sum(np.sum(J[:,:,None,:]*mass_matrix, axis=-1)*J, axis=(-1,-2))

            k = m_eff * (self.w**2)
            c = m_eff * 2*self.zeta*self.w

            CFM = 1./(c+dt*k)

            ERP = dt*k/(c+dt*k)
            m_c = 1./(1./m_eff + CFM)
            b = ERP/dt * b_error + b_res

            lamb = - m_c[:] * (np.sum(J[:,:,:]*v[:,:,:], axis=(1,2)) + CFM * self.P[:] + b)

            self.P += lamb

            clipping_limit = np.abs(self.clipping_a * self.P[self.clipping_idx] + self.clipping_b * dt)
            self.P = np.clip(self.P,-clipping_limit, clipping_limit)
            applicable = (1-(self.only_when_positive*( 1-(self.P>=0)) )) * self.C


            result = np.sum(mass_matrix*J[:,:,None,:], axis=(-1)) * self.P[:,None,None] * applicable[:,None,None]
            result = result.reshape(result.shape[:-3] + (2*self.num_constraints,6))

            for i in xrange(newv.shape[0]):
                newv[i,:] = newv[i,:] + np.sum(result[self.map_object_to_constraint[i],:], axis=0)
                #newv[i,:] = originalv[i,:] + np.sum(result[self.map_object_to_constraint[i],:], axis=0)

        #print
        return newv


    def do_time_step(self, dt=None, motor_signals=list()):
        if dt is None:
            dt = self.DT

        ##################
        # --- Step 3 --- #
        ##################
        # In the third step, we integrate the new position x2 of the bodies using the new velocities
        # v2 computed in the second step with : x2 = x1 + dt * v2.

        # semi-implicit Euler integration
        self.velocityVectors = self.evaluate(dt, self.positionVectors, self.velocityVectors, motor_signals=motor_signals)
        self.positionVectors[:,:3] = self.positionVectors[:,:3] + self.velocityVectors[:,:3] * dt

        self.rot_matrices[:,:,:] = self.normalize_matrix(self.rot_matrices[:,:,:] + np.sum(self.rot_matrices[:,:,:,None] * skew_symmetric(dt * self.velocityVectors[:,3:])[:,None,:,:],axis=2) )

    def normalize_matrix(self, A):
        for i in xrange(self.rotation_reorthogonalization_iterations):
            A = (3*A - np.sum(A[...,:,None,:,None] * A[...,None,:,:,None] * A[...,None,:,None,:], axis=(-3,-2))) / 2
        return A

    def getState(self):
        return self.positionVectors[:,:3], self.velocityVectors, self.rot_matrices

    def getObjectIndex(self, reference):
        return self.objects[reference]




    def getPosition(self, reference):
        idx = self.objects[reference]
        return self.positionVectors[idx]

    def getRotationMatrix(self, reference):
        idx = self.objects[reference]
        return self.rot_matrices[idx]

    def getSensorValues(self, reference):
        # make positionvectors neutral according to reference object
        idx = self.objects[reference]
        rot_matrix = self.rot_matrices[idx]

        ref_position = self.positionVectors[idx,:]

        result = self.positionVectors
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


