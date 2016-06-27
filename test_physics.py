from direct.showbase.ShowBase import ShowBase
from math import pi, sin, cos
from direct.task import Task
from panda3d.core import Point2, Texture, CardMaker, AmbientLight, Vec4, DirectionalLight, Spotlight, Quat
from PhysicsSystem import Rigid3DBodyEngine
import time
import json

def fixQuat(quat):

    quat = (-quat[0],quat[1],quat[2],quat[3])
    return Quat(*quat)

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.names = []
        self.setFrameRateMeter(True)
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        cm.setUvRange(Point2(-2000/5,-2000/5),Point2(2000/5,2000/5))
        tmp = self.render.attachNewNode(cm.generate())
        tmp.reparentTo(self.render)

        tmp.setPos(0, 0, 0)
        tmp.lookAt((0, 0, -2))
        tmp.setColor(1.0,1.0,1.0,0.)
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
        directionalLightNP.setHpr(-130, -50, 0)
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
        self.load_robot_model("robotmodel/predator.json")

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
        self.physics.addSphere(name, radius, position+rotation, velocity)


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
        self.physics.addCube(name, dimensions, position + rotation, velocity)

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
            parameters = dict(robot_dict["default_joint_parameters"]["default"])  # copy
            if joint["type"] in robot_dict["default_joint_parameters"]:
                parameters.update(robot_dict["default_joint_parameters"][joint["type"]])
            parameters.update(joint)
            if joint["type"] == "hinge":
                self.physics.addHingeConstraint(jointname, **parameters)

            elif joint["type"] == "ground":
                self.physics.addGroundConstraint(jointname, **parameters)

            elif joint["type"] == "fixed":
                self.physics.addFixedConstraint(jointname, **parameters)

            elif joint["type"] == "ball":
                self.physics.addBallAndSocketConstraint(jointname, **parameters)
        """
        for motorname, motor in robot_dict["motors"].iteritems():
            parameters = dict(robot_dict["default_motor_parameters"]["default"])  # copy
            if joint["type"] in robot_dict["default_motor_parameters"]:
                parameters.update(robot_dict["default_motor_parameters"][joint["type"]])
            parameters.update(joint)
        """



    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        self.physics.do_time_step(dt=0.005)
        for obj_name, obj in self.objects.iteritems():
            if (abs(self.physics.getPosition(obj_name)) > 10**5).any():
                print "problem with", obj_name
            obj.setPos(*self.physics.getPosition(obj_name)[:3])
            obj.setQuat(self.render, fixQuat(self.physics.getPosition(obj_name)[3:]))

        # change camera movement
        self.camera.setPos(0,20,3)
        self.camera.lookAt(0,0,3)
        time.sleep(0.001)
        return Task.cont

app = MyApp()
app.run()