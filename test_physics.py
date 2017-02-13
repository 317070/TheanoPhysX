from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from math import pi, sin, cos
from direct.task import Task
from panda3d.core import Point2, Texture, CardMaker, AmbientLight, Vec4, DirectionalLight, Spotlight, Quat, LMatrix4f, \
    LMatrix3f, TextureStage, WindowProperties
from PhysicsSystem import Rigid3DBodyEngine
import time
import json
import numpy as np


def fixQuat(quat):

    quat = (-quat[0],quat[1],quat[2],quat[3])
    return Quat(*quat)

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        props = WindowProperties( )
        props.setTitle( 'Differentiable Physics Engine' )
        self.win.requestProperties( props )
        self.t = 0
        self.starttime = time.time()
        #self.setFrameRateMeter(True)
        cour = self.loader.loadFont('cmtt12.egg')
        self.textObject = OnscreenText(font= cour, text = 'abcdefghijklmnopqrstuvwxyz', pos=(0, -0.045), parent = self.a2dTopCenter, bg=(0,0,0,0.5), fg =(1,1,1,1), scale = 0.07, mayChange=True)
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        cm.setUvRange(Point2(-2000/5,-2000/5),Point2(2000/5,2000/5))

        tmp = self.render.attachNewNode(cm.generate())
        tmp.reparentTo(self.render)
        self.camLens.setNear(0.1)

        tmp.setPos(0, 0, 0)
        tmp.lookAt((0, 0, -2))
        tmp.setColor(1.0,1.0,1.0,0.)
        tmp.setTexScale(TextureStage.getDefault(), 1, 1)
        tex = self.loader.loadTexture('textures/grid2.png')

        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        tmp.setTexture(tex,1)
        self.setBackgroundColor(0.0, 191.0/255.0, 1.0, 1.0) #color of the sky

        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # Directional light 01
        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        # This light is facing backwards, towards the camera.
        directionalLightNP.setHpr(-120, -50, 0)
        directionalLightNP.node().setScene(self.render)
        directionalLightNP.node().setShadowCaster(True)
        directionalLightNP.node().getLens().setFov(40)
        directionalLightNP.node().getLens().setNearFar(10, 100)
        self.render.setLight(directionalLightNP)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")


        self.physics = Rigid3DBodyEngine()
        # Load the environment model.
        self.objects = dict()

        #self.load_robot_model("robotmodel/test.json")
        #self.load_robot_model("robotmodel/predator.json")
        #self.load_robot_model("robotmodel/full_predator.json")
        #self.load_robot_model("robotmodel/demi_predator.json")
        # self.load_robot_model("robotmodel/demi_predator_ground.json")
        self.load_robot_model("robotmodel/car.json")
        #self.load_robot_model("robotmodel/ball.json")
        #self.load_robot_model("robotmodel/robot_arm.json")
        #self.load_robot_model("robotmodel/robot_arm_mini.json")
        self.physics.compile()
        self.step = np.zeros(shape=(16,))
        self.state = self.physics.get_initial_state()

    def run_no_gui(self):
        while True:
            DT = 0.01
            ph = self.t*2*np.pi
            self.physics.do_time_step(dt=DT, motor_signals=[0,0,0,0]+
                                                           [1,0,0,0,0,0,0,0,0,0,0,0])

            self.t += DT
            real_time = time.time() - self.starttime
            #
            if real_time>10:
                break
        print self.t/real_time
        self.userExit()


    def add_sphere(self, name, radius, position, rotation, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        smiley = self.loader.loadModel("smiley")
        smiley.setScale(radius,radius,radius)
        smiley.setTexture(self.loader.loadTexture('textures/soccer.png'), 1)
        #smiley.setColor(0,0,0.1)

        # Reparent the model to render.
        smiley.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        smiley.setPos(*position)
        smiley.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = smiley


    def add_cube(self, name, dimensions, position, rotation, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        cube = self.loader.loadModel("textures/car.egg")
        cube.setScale(*dimensions)
        # cube.setTexture(self.loader.loadTexture('maps/noise.rgb'), 1)

        # tex = self.loader.loadTexture('textures/tesla_128.png')
        # tex.setWrapU(Texture.WMClamp)
        # tex.setWrapV(Texture.WMClamp)
        # cube.setTexture(tex,1)
        # if "color" in parameters:
        #     cube.setColor(*parameters["color"])

        # Reparent the model to render.
        cube.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        cube.setPos(*position)
        cube.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = cube



    def load_robot_model(self, filename):
        self.physics.load_robot_model(filename)
        robot_dict = json.load(open(filename, "rb"))

        for elementname, element in robot_dict["model"].iteritems():
            primitive = element[0]
            parameters = dict(robot_dict["default_model_parameters"]["default"])  # copy
            if primitive["shape"] in robot_dict["default_model_parameters"]:
                parameters.update(robot_dict["default_model_parameters"][primitive["shape"]])
            parameters.update(primitive)
            if primitive["shape"] == "cube":
                self.add_cube(elementname, **parameters)
            elif primitive["shape"] == "sphere":
                self.add_sphere(elementname, **parameters)
            # TODO
            # elif primitive["shape"] == "plane":
            #     self.add_plane(elementname, **parameters)
            # elif primitive["shape"] == "face":
            #     self.add_face(**parameters)




    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        self.t += self.physics.DT
        ph = self.t*np.float32(2*np.pi/2.5)
        #sensors = self.physics.get_sensor_values("spine").flatten()
        #print sensors.shape
        import math
        self.state = self.physics.do_time_step(self.state, motor_signals=[math.sin(ph),math.sin(ph),2,2])

        positions, velocity, rotations = self.state
        for obj_name, obj in self.objects.iteritems():
            obj_id = self.physics.get_object_index(obj_name)
            if (abs(positions[obj_id,:]) > 10**5).any():
                print "problem with", obj_name
            sc = obj.getScale()

            #print obj_name, self.physics.getRotationMatrix(obj_name).flatten()
            obj.setMat(self.render, LMatrix4f(LMatrix3f(*rotations[obj_id,:,:].flatten())))
            obj.setPos(*positions[obj_id,:])
            obj.setScale(sc)

        # change camera movement
        self.camera.setPos(-10,0,1.5)
        self.camera.lookAt(0,0,0)
        # self.camera.lookAt(*self.physics.getPosition(self.physics.camera_focus)[:3])
        #print self.t, self.physics.getPosition(self.physics.camera_focus)
        real_time = time.time() - self.starttime

        self.textObject.setText('Time: %3.3f s\n%3.3fx real time\n%s' % ( self.t, self.t/real_time , ""))
        time.sleep(0.0001)
        if self.t>80:
            self.userExit()
        return Task.cont



app = MyApp()
import cProfile
import re
#cProfile.run('app.run_no_gui()')
app.run()

