import cPickle as pickle
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
        self.textObject = OnscreenText(font= cour, text = 'abcdefghijklmnopqrstuvwxyz', pos=(0, -0.045), parent = self.a2dTopCenter, bg=(0,0,0,1), fg =(1,1,1,1), scale = 0.07, mayChange=True)
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        cm.setUvRange(Point2(-2000/5,-2000/5),Point2(2000/5,2000/5))

        tmp = self.render.attachNewNode(cm.generate())
        tmp.reparentTo(self.render)

        tmp.setPos(0, 0, 0)
        tmp.lookAt((0, 0, -2))
        tmp.setColor(1.0,1.0,1.0,1)
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
        directionalLightNP.setHpr(-60, -50, 0)
        directionalLightNP.node().setScene(self.render)
        directionalLightNP.node().setShadowCaster(True)
        directionalLightNP.node().getLens().setFov(40)
        directionalLightNP.node().getLens().setNearFar(10, 100)
        self.render.setLight(directionalLightNP)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")


        # Load the environment model.
        self.objects = dict()
        self.names = []
        data = pickle.load(open("../PhysXVids/state-dump-exp15.pkl","rb"))

        self.json = json.loads(data["json"]) # json.loads(data["json"])
        self.states = data["states"]
        self.load_robot_model()
        self.dt = self.json["integration_parameters"]["time_step"]
        self.setupKeys()
        self.robot_id = 0
        #self.movie(duration = 8.0)
        #self.target = data["targets"]

    def setupKeys(self):
        self.parentnode = self.render.attachNewNode('camparent')
        self.parentnode.setPos(0,0,0.5)
        self.parentnode.setH(0)
        self.parentnode.setP(0)
        self.parentnode.setR(0)
        rotnode = self.parentnode.attachNewNode('rotnode')
        rotnode.setH(0)
        rotnode.setP(0)
        rotnode.setR(0)
        rotnode.setPos(0,-3.5,0)
        #self.camera.lookAt(self.objects[self.camera_focus])
        self.camera.wrtReparentTo(rotnode)
        self.camLens.setNear(0.1)

        keyMap = {"arrow_left":0, "arrow_right":0, "arrow_up":0, "arrow_down":0, "i":0, "u":0,  "mouse1":0, "usemouse":False, "click":False}
        def setKey(key, value):
           keyMap[key] = value

        self.accept('arrow_left', setKey, ["arrow_left",1])
        self.accept('arrow_left-up', setKey, ["arrow_left",0])
        self.accept('arrow_right', setKey, ["arrow_right",1])
        self.accept('arrow_right-up', setKey, ["arrow_right",0])
        self.accept('arrow_up', setKey, ["arrow_up",1])
        self.accept('arrow_up-up', setKey, ["arrow_up",0])
        self.accept('arrow_down', setKey, ["arrow_down",1])
        self.accept('arrow_down-up', setKey, ["arrow_down",0])
        self.accept('+', setKey, ["i",1])
        self.accept('+-up', setKey, ["i",0])
        self.accept('-', setKey, ["u",1])
        self.accept('--up', setKey, ["u",0])
        self.accept('mouse1', setKey, ["mouse1",1])
        self.accept('mouse1-up', setKey, ["mouse1",0])


        def cameraMovement(task):
            self.disableMouse()
            if keyMap["arrow_left"]!=0:
                self.parentnode.setH(self.parentnode.getH()+80 * 0.01)
            if keyMap["arrow_right"]!=0:
                self.parentnode.setH(self.parentnode.getH()-80 * 0.01)
            if keyMap["arrow_up"]!=0:
                self.parentnode.setP(self.parentnode.getP()+60 * 0.01)
            if keyMap["arrow_down"]!=0:
                self.parentnode.setP(self.parentnode.getP()-60 * 0.01)
            if keyMap["u"]!=0:
                self.robot_id -= 1
            if keyMap["i"]!=0:
                self.robot_id += 1

            if keyMap["mouse1"]!=0:
                if not keyMap["click"]:
                    keyMap["click"]=True
                    if keyMap["usemouse"]:
                        #stop using the mouse
                        keyMap["usemouse"] = False
                        props = WindowProperties()
                        props.setCursorHidden(False)
                        self.win.requestProperties(props)
                    else:
                        keyMap["usemouse"] = True
                        props = WindowProperties()
                        props.setCursorHidden(True)
                        self.win.requestProperties(props)
            else:
                keyMap["click"]=False

            if keyMap["usemouse"]:

                md = self.win.getPointer(0)
                x = md.getX()
                y = md.getY()
                if self.win.movePointer(0, 300, 300):
                    if not keyMap["click"]:

                        self.parentnode.setH(self.parentnode.getH() - (x-300)*0.5)
                        self.parentnode.setP(self.parentnode.getP() - (y-300)*0.5)
            return task.cont

        self.taskMgr.add( cameraMovement, 'cameraMovement')



    def addSphere(self, name, radius, mass_density, position, rotation, velocity, **parameters):
        #smiley = self.loader.loadModel("zup-axis")
        smiley = self.loader.loadModel("smiley")
        smiley.setScale(radius,radius,radius)
        #smiley.setTexture(self.loader.loadTexture('textures/soccer.png'), 1)
        smiley.setTextureOff(1)
        smiley.setColor(0.2,0.2,0.8)

        # Reparent the model to render.
        smiley.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        smiley.setPos(*position)
        smiley.setQuat(self.render, fixQuat(rotation))

        self.objects[name] = smiley
        self.names.append(name)

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
        self.names.append(name)


    def load_robot_model(self):
        robot_dict = self.json
        #self.camera_focus = robot_dict["camera_focus"]

        if "universe" in robot_dict["integration_parameters"] and robot_dict["integration_parameters"]["universe"]:
            self.names.append("universe")

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


        self.addSphere("target", radius=0.01, mass_density=0, position=[0,0,0], rotation=[1,0,0,0], velocity=[0,0,0,0,0,0])
        self.names.remove("target")


    # Define a procedure to move the camera.
    def spinCameraTask(self, task):

        frame_step = 0.01
        self.t += frame_step

        positions, velocities, rotations = self.states[0], self.states[1], self.states[2]
        step = int(self.t / self.dt)
        step = step % positions.shape[0]
        self.t = self.t % (self.dt*positions.shape[0])
        robot_id = self.robot_id % positions.shape[1]
        print np.linalg.norm(velocities[step,robot_id,0,:3])
        for idx, name in enumerate(self.names):
            if name in self.objects:
                obj = self.objects[name]
                sc = obj.getScale()
                #print obj_name, self.physics.getRotationMatrix(obj_name).flatten()

                if np.isfinite(positions[step,robot_id,idx]).all() and np.isfinite(rotations[step,robot_id,idx]).all():
                    obj.setMat(self.render, LMatrix4f(LMatrix3f(*rotations[step,robot_id,idx].flatten())))
                    obj.setPos(*positions[step,robot_id,idx])
                    obj.setScale(sc)

        #print np.sqrt(np.sum((positions[step,robot_id,self.names.index("sphere2"),:]-np.array([0.5,0.5,0.5]))**2))
        #if np.isfinite(positions[step,robot_id,self.names.index(self.camera_focus),:]).all():
        #    self.parentnode.setX(positions[step,robot_id,self.names.index(self.camera_focus),0])
        #    self.parentnode.setY(positions[step,robot_id,self.names.index(self.camera_focus),1])

        #print self.names
        # change camera movement
        #self.camera.setPos(1.5,3.5,1.5)
        #if np.isfinite(positions[step,robot_id,self.names.index(self.camera_focus)]).all():
        #    self.camera.lookAt(*positions[step,robot_id,self.names.index(self.camera_focus)])
        #self.objects["target"].setPos(*self.target[robot_id,:])
        #print self.t, self.physics.getPosition("ball")
        #real_time = time.time() - self.starttime

        #self.textObject.setText('Time: %3.3f s\nVx: %3.3f\nrobot #%d' % ( self.t, velocities[step,robot_id,self.names.index(self.camera_focus),0], robot_id))
        time.sleep(frame_step)
        return Task.cont



app = MyApp()
import cProfile
import re
#cProfile.run('app.run_no_gui()')
app.run()

