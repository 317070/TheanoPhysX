from direct.showbase.ShowBase import ShowBase
from math import pi, sin, cos
from direct.task import Task
from panda3d.core import Point2, Texture, CardMaker, AmbientLight, Vec4, DirectionalLight, Spotlight, Quat
from PhysicsSystem2 import Rigid3DBodyEngine
import time


def fixQuat(quat):

    quat = (-quat[0],quat[1],quat[2],quat[3])
    return Quat(*quat)

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.setFrameRateMeter(True)
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        cm.setUvRange(Point2(-2000/5,-2000/5),Point2(2000/5,2000/5))
        tmp = self.render.attachNewNode(cm.generate())
        tmp.reparentTo(self.render)

        tmp.setPos(0, 0, -1)
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
        directionalLightNP.node().showFrustum()
        directionalLightNP.node().getLens().setFov(40)
        directionalLightNP.node().getLens().setNearFar(10, 100)
        self.render.setLight(directionalLightNP)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")


        self.physics = Rigid3DBodyEngine()
        # Load the environment model.
        self.objects = list()
        def addSphere(position, velocity):
            #smiley = self.loader.loadModel("zup-axis")
            smiley = self.loader.loadModel("smiley")
            #smiley.setScale(0.2,0.2,0.2)
            smiley.setTexture(self.loader.loadTexture('maps/noise.rgb'), 1)

            # Reparent the model to render.
            smiley.reparentTo(self.render)
            # Apply scale and position transforms on the model.
            smiley.setPos(*position[:3])
            smiley.setQuat(self.render, fixQuat(position[3:]))

            self.objects.append(smiley)
            self.physics.addSphere(smiley, position, velocity)
            self.physics.addConstraint("ground",[smiley],{"mu":0.2, "alpha":0.6, "gamma":0.0, "delta":0.001, "torsional_friction": False})


        addSphere([0,0,0,0,1,0,0], [-6,0,6,0,0,0])
        addSphere([0,0,2,0,1,0,0], [0,0,0,0,0,0])

        #self.physics.addBallAndSocketConstraint(self.objects[0], self.objects[1],[0,0,1],{"beta": 0.8})

        self.physics.addHingeConstraint(self.objects[0], self.objects[1],[0,0,1],[0,1,0], {"beta": 0.001, "motor_position": 1, "motor_velocity": 1, "motor_torque": 0.5, "delta":0.01})



    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        self.physics.do_time_step(dt=5e-3)
        for obj in self.objects:
            obj.setPos(*self.physics.getPosition(obj)[:3])
            obj.setQuat(self.render, fixQuat(self.physics.getPosition(obj)[3:]))

        # change camera movement
        angleDegrees = task.time * 3.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20.0 * cos(angleRadians), 3)
        self.camera.lookAt(*self.physics.getPosition(self.objects[0])[:3])
        time.sleep(0.001)
        return Task.cont

app = MyApp()
app.run()