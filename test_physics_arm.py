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
        self.textObject = None#OnscreenText(font= cour, text = 'abcdefghijklmnopqrstuvwxyz', pos=(0, -0.045), parent = self.a2dTopCenter, bg=(0,0,0,0.5), fg =(1,1,1,1), scale = 0.07, mayChange=True)
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
        #self.load_robot_model("robotmodel/ball.json")
        self.load_robot_model("robotmodel/robot_arm.json")
        #self.load_robot_model("robotmodel/robot_arm_mini.json")
        self.physics.compile()
        self.step = np.zeros(shape=(16,))

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


    def addSphere(self, name, radius, mass_density, position, rotation, velocity, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        smiley = self.loader.loadModel("smiley")
        smiley.setScale(radius,radius,radius)
        #smiley.setTexture(self.loader.loadTexture('textures/soccer.png'), 1)
        smiley.setColor(0,0,0.5)

        # Reparent the model to render.
        smiley.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        smiley.setPos(*position)
        smiley.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = smiley
        self.physics.add_sphere(name, radius, mass_density, position+rotation, velocity)


    def addCube(self, name, dimensions, mass_density, position, rotation, velocity, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        cube = self.loader.loadModel("textures/box.egg")
        cube.setScale(*dimensions)
        cube.setTexture(self.loader.loadTexture('maps/noise.rgb'), 1)

        tex = self.loader.loadTexture('textures/square.png')
        tex.setWrapU(Texture.WMClamp)
        tex.setWrapV(Texture.WMClamp)
        cube.setTexture(tex,1)
        if "color" in parameters:
            cube.setColor(*parameters["color"])

        # Reparent the model to render.
        cube.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        cube.setPos(*position)
        cube.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = cube
        self.physics.add_cube(name, dimensions, mass_density, position + rotation, velocity)


    def load_robot_model(self, filename):
        robot_dict = json.load(open(filename,"rb"))
        self.physics.camera_focus = robot_dict["camera_focus"]
        self.physics.set_integration_parameters(**robot_dict["integration_parameters"])

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
                self.physics.add_hinge_constraint(jointname, **parameters)

            elif joint["type"] == "ground":
                self.physics.add_ground_constraint(jointname, **parameters)

            elif joint["type"] == "fixed":
                self.physics.add_fixed_constraint(jointname, **parameters)

            elif joint["type"] == "ball":
                self.physics.add_ball_and_socket_constraint(jointname, **parameters)

            if "limits" in parameters:
                for limit in parameters["limits"]:
                    limitparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "limit" in robot_dict["default_constraint_parameters"]:
                        limitparameters.update(robot_dict["default_constraint_parameters"]["limit"])
                    limitparameters.update(limit)
                    self.physics.addLimitConstraint(joint["object1"], joint["object2"], **limitparameters)

            #"""
            if "motors" in parameters:
                for motor in parameters["motors"]:
                    motorparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "motor" in robot_dict["default_constraint_parameters"]:
                        motorparameters.update(robot_dict["default_constraint_parameters"]["motor"])
                    motorparameters.update(motor)
                    self.physics.addMotorConstraint(joint["object1"], joint["object2"], **motorparameters)
            #"""



    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        self.t += self.physics.DT
        ph = self.t*np.float32(2*np.pi*1.5)/15
        #sensors = self.physics.get_sensor_values("spine").flatten()
        #print sensors.shape
        #self.physics.do_time_step(motor_signals=[-sin(ph),sin(ph),-1,1,0,0,0,0,0,0,0,0,0,0,0,0])
        ALPHA = 1.00
        self.step = (1-ALPHA) * self.step + ALPHA*np.random.randn(16)*30
        A1, A2, A3, A4, B1, B2, B3, B4 = 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0, 0
        #self.physics.do_time_step(motor_signals=[-A1*sin(ph)+B1,-A1*sin(ph)+B1,-A2*sin(ph)-B2,-A2*sin(-ph)-B2,-A3*cos(ph)+B3,A3*cos(ph)+B3,A4*cos(ph)+B4,-A4*cos(ph)+B4])
        #self.physics.do_time_step(motor_signals=[A1*sin(ph)+B1,-A1*sin(ph)+B1,-A2*sin(ph)+B2,A2*sin(ph)+B2,A3*cos(ph)+B3,-A3*cos(ph)+B3,-A4*cos(ph)+B4,A4*cos(ph)+B4])
        p4 = np.pi/4
        p3 = 3.*np.pi/4.
        p2 = np.pi/2
        p1 = np.pi
        self.physics.do_time_step(motor_signals=np.array([A1*sin(ph)+B1,A1*sin(ph)+B1,-A2*sin(ph)-B2,-A2*sin(-ph)-B2], dtype='float32'))
        #self.physics.do_time_step(motor_signals=[-p2,-p4,0,0])


        for obj_name, obj in self.objects.iteritems():
            if (abs(self.physics.getPosition(obj_name)) > 10**5).any():
                print "problem with", obj_name
            sc = obj.getScale()

            #print obj_name, self.physics.getRotationMatrix(obj_name).flatten()
            obj.setMat(self.render, LMatrix4f(LMatrix3f(*self.physics.getRotationMatrix(obj_name).flatten())))
            obj.setPos(*self.physics.getPosition(obj_name)[:3])
            obj.setScale(sc)

        # change camera movement
        self.camera.setPos(1.5,3.5,1.5)
        #self.camera.lookAt(0,0,3)
        self.camera.lookAt(*self.physics.getPosition(self.physics.camera_focus)[:3])
        #print self.t, self.physics.getPosition(self.physics.camera_focus)
        real_time = time.time() - self.starttime

        if self.textObject:
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

