from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from math import pi, sin, cos
from direct.task import Task
from panda3d.core import Point2, Texture, CardMaker, AmbientLight, Vec4, DirectionalLight, Spotlight, Quat, LMatrix4f, \
    LMatrix3f, TextureStage
from PhysicsSystem import Rigid3DBodyEngine
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
import time
import json
import numpy as np
import theano

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def fixQuat(quat):
    quat = (-quat[0],quat[1],quat[2],quat[3])
    return Quat(*quat)

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        self.t = 0
        self.starttime = None
        self.setFrameRateMeter(True)
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
        #tmp.setTexScale(TextureStage.getDefault(), 1, 1)
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

        self.physics = TheanoRigid3DBodyEngine()
        self.benchmark_physics = None#Rigid3DBodyEngine()
        # Load the environment model.
        self.objects = dict()

        #self.load_robot_model("robotmodel/test.json")
        self.load_robot_model("robotmodel/predator.json")
        #self.load_robot_model("robotmodel/simple_predator.json")
        if self.benchmark_physics:
            self.benchmark_physics.compile()


        print "Compiling..."
        self.positions, self.velocities, self.rotations = self.physics.getInitialState()
        self.physics.compile()

        import theano.tensor as T
        positions = T.fmatrix()
        velocities = T.fmatrix()
        rotations =  T.ftensor3()

        a,b,c = self.physics.step_from_this_state((positions,velocities,rotations), dt=0.001, motor_signals=[-1,1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0])
        self.timestep = theano.function(inputs=[positions, velocities, rotations], outputs=[a,b,c], allow_input_downcast=True)



    def addSphere(self, name, radius, position, rotation, velocity, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        smiley = self.loader.loadModel("smiley")
        smiley.setScale(radius,radius,radius)
        smiley.setTexture(self.loader.loadTexture('textures/soccer.png'), 1)

        # Reparent the model to render.
        smiley.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        smiley.setPos(*position)
        smiley.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = smiley
        self.physics.addSphere(name, radius, position, rotation, velocity)
        if self.benchmark_physics:
            self.benchmark_physics.addSphere(name, radius, position + rotation, velocity)


    def addCube(self, name, dimensions, position, rotation, velocity, **parameters):
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
        self.physics.addCube(name, dimensions, position, rotation, velocity)
        if self.benchmark_physics:
            self.benchmark_physics.addCube(name, dimensions, position + rotation, velocity)



    def load_robot_model(self, filename):
        robot_dict = json.load(open(filename,"rb"))
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
                self.physics.addHingeConstraint(jointname, **parameters)
                if self.benchmark_physics:
                    self.benchmark_physics.addHingeConstraint(jointname, **parameters)

            elif joint["type"] == "ground":
                self.physics.addGroundConstraint(jointname, **parameters)
                if self.benchmark_physics:
                    self.benchmark_physics.addGroundConstraint(jointname, **parameters)

            elif joint["type"] == "fixed":
                self.physics.addFixedConstraint(jointname, **parameters)
                if self.benchmark_physics:
                    self.benchmark_physics.addFixedConstraint(jointname, **parameters)

            elif joint["type"] == "ball":
                self.physics.addBallAndSocketConstraint(jointname, **parameters)
                if self.benchmark_physics:
                    self.benchmark_physics.addBallAndSocketConstraint(jointname, **parameters)

            if "limits" in parameters:
                for limit in parameters["limits"]:
                    limitparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "limit" in robot_dict["default_constraint_parameters"]:
                        limitparameters.update(robot_dict["default_constraint_parameters"]["limit"])
                    limitparameters.update(limit)
                    self.physics.addLimitConstraint(joint["object1"], joint["object2"], **limitparameters)
                    if self.benchmark_physics:
                        self.benchmark_physics.addLimitConstraint(joint["object1"], joint["object2"], **limitparameters)

            #"""
            if "motors" in parameters:
                for motor in parameters["motors"]:
                    motorparameters = dict(robot_dict["default_constraint_parameters"]["default"])
                    if "motor" in robot_dict["default_constraint_parameters"]:
                        motorparameters.update(robot_dict["default_constraint_parameters"]["motor"])
                    motorparameters.update(motor)
                    self.physics.addMotorConstraint(joint["object1"], joint["object2"], **motorparameters)
                    if self.benchmark_physics:
                        self.benchmark_physics.addMotorConstraint(joint["object1"], joint["object2"], **motorparameters)
            #"""



    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        if self.starttime is None:
            self.starttime = time.time()
        DT = 0.01
        self.t += DT

        self.positions, self.velocities, self.rotations = self.timestep(self.positions, self.velocities, self.rotations)
        #print positions, rotations, velocity

        for obj_name, obj in self.objects.iteritems():
            sc = obj.getScale()

            idx = self.physics.getObjectIndex(obj_name)
            obj.setMat(self.render, LMatrix4f(LMatrix3f(*self.rotations[idx,:,:].flatten())))
            obj.setPos(*self.positions[idx,:])
            obj.setScale(sc)

        # change camera movement
        self.camera.setPos(0,2,0.3)
        #self.camera.lookAt(0,0,3)
        self.camera.lookAt(*self.positions[self.physics.getObjectIndex("spine"),:])

        real_time = time.time() - self.starttime

        self.textObject.setText('Time: %3.3f s\n%3.3fx real time\n%s' % ( self.t, self.t/real_time , ""))
        #time.sleep(0.001)
        if real_time>100:
            self.userExit()

        return Task.cont



app = MyApp()
import cProfile
import re
#cProfile.run('app.run()')
app.run()

