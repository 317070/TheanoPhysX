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
        directionalLightNP.setHpr(0, -50, 0)
        directionalLightNP.node().setScene(self.render)
        directionalLightNP.node().setShadowCaster(True)
        directionalLightNP.node().getLens().setFov(40)
        directionalLightNP.node().getLens().setNearFar(10, 100)
        self.render.setLight(directionalLightNP)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")


        self.physics = Rigid3DBodyEngine()
        # Load the environment model.
        self.objects = list()


        self.addSphere(0, 2, [0,0,4,1,0,0,0], [-1,0,0,0,0,2])
        self.addSphere(1, 1, [0,0,1,1,0,0,0], [0,0,0,0,-0,2])

        #self.physics.addBallAndSocketConstraint(self.objects[0], self.objects[1],[0,0,1],{"beta": 0.8})
        #self.physics.addSliderConstraint(self.objects[0], self.objects[1],{"beta": 0.8})
        self.physics.addFixedConstraint(self.objects[0], self.objects[1],[0,0,2],{"beta": 0.8})

        #self.physics.addHingeConstraint(self.objects[0], self.objects[1],[0,0,1],[0,1,0], {"beta": 0.001, "motor_position": -1, "motor_velocity": 1, "motor_torque": 0.5, "delta":0.01})
        self.load_robot_model("robotmodel/test.json")

    def addSphere(self, name, radius, position, velocity):
        #smiley = self.loader.loadModel("zup-axis")
        smiley = self.loader.loadModel("smiley")
        smiley.setScale(radius,radius,radius)
        smiley.setTexture(self.loader.loadTexture('maps/noise.rgb'), 1)

        # Reparent the model to render.
        smiley.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        smiley.setPos(*position[:3])
        smiley.setQuat(self.render, fixQuat(position[3:]))

        self.objects.append(smiley)
        self.physics.addSphere(smiley, radius, position, velocity)
        self.physics.addGroundConstraint(smiley,{"mu":0.5, "alpha":0.6, "gamma":0.0, "delta":0.001, "torsional_friction": True})


    def addCube(self, name, sizes, position, velocity):
        #smiley = self.loader.loadModel("zup-axis")
        cube = self.loader.loadModel("box")
        cube.setScale(*sizes)
        cube.setTexture(self.loader.loadTexture('maps/noise.rgb'), 1)

        # Reparent the model to render.
        cube.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        cube.setPos(*position[:3])
        cube.setQuat(self.render, fixQuat(position[3:]))

        self.objects.append(cube)
        self.physics.addCube(cube, sizes, position, velocity)

    def load_robot_model(self, filename):
        robot_dict = json.load(open(filename,"rb"))
        for elementname, element in robot_dict["model"].iteritems():
            primitive = element[0]
            print primitive
            if primitive["shape"] == "cube":
                self.addCube(elementname, primitive["dimensions"], primitive["position"]+primitive["rotation"],[0,0,0,0,0,0])
            if primitive["shape"] == "sphere":
                self.addCube(elementname, primitive["dimensions"], primitive["position"]+primitive["rotation"],[0,0,0,0,0,0])

        for jointname, joint in robot_dict["joints"].iteritems():
            if joint["type"] == "hinge":
                self.physics.addHingeConstraint(joint["from"], joint["to"], joint["point"], joint["axis"], {"beta": 0.001, "motor_position": -1, "motor_velocity": 1, "motor_torque": 0.5, "delta":0.01})

            if joint["type"] == "ground":
                self.physics.addGroundConstraint(joint["from"], {"mu":0.5, "alpha":0.6, "gamma":0.0, "delta":0.001, "torsional_friction": True})





    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        self.physics.do_time_step(dt=5e-3)
        for obj in self.objects:
            obj.setPos(*self.physics.getPosition(obj)[:3])
            obj.setQuat(self.render, fixQuat(self.physics.getPosition(obj)[3:]))

        # change camera movement
        self.camera.setPos(*(self.physics.getPosition(self.objects[0])[:3] - [0,20,0]))
        self.camera.lookAt(*self.physics.getPosition(self.objects[0])[:3])
        time.sleep(0.001)
        return Task.cont

app = MyApp()
app.run()