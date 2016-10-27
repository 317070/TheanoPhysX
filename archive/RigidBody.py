

class RigidBody(object):
    def __init__(self):
        self.primitives = []

    def addPrimitiveObject(self, primitive, location, orientation):
        pass

    def getCenterOfMass(self):
        pass

    def getInertiaMatrix(self):
        pass

    def getPrimitiveObjectState(self):
        pass

class Primitive(object):
    def __init__(self, state):
        self.state = state

    def getCenterOfMass(self):
        pass

    def getInertiaMatrix(self):
        pass

    def addSelfToRender(self, render):
        pass



class Cube(Primitive):
    pass

class Sphere(Primitive):
    pass