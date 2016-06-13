import os#GEN
import sys#GEN
from pandac.PandaModules import OdeMass, LineSegs, OdeAMotorJoint, OdeLMotorJoint, OdeHingeJoint, OdeSliderJoint, CardMaker, BitMask32, Quat, OdeJointGroup, OdeSimpleSpace, Mat3, Vec4, Vec3, OdeWorld,OdeBody, OdeBoxGeom, OdeContactGeom, OdeContactJoint, OdeCylinderGeom, OdeFixedJoint, OdeGeom, OdeSpace, OdeJoint, OdeSphereGeom, OdeBoxGeom, OdePlaneGeom#GEN
import bin.aux_types as aux#GEN
import cPickle as pickle#GEN
from numpy import *#GEN
class PhysicsModel(object):#GEN
	def __init__(self):	#GEN
		#load the ODE tree		#GEN
				#GEN
		for folder in sys.path:
			if 'robotmodel' in folder:
				self.ode_tree = pickle.load(open(folder+'/abstract-tree','rb'))			#GEN
				break
		#general properties		#GEN
		#GEN
		self.ode_timestep = 0.002		#GEN
		self.ode_dt = self.ode_timestep		#GEN
		self.totaltime = 0.000		#GEN
		#print 'ODE_DT: %g'%self.ode_dt		#GEN
		self.ode_northDirection = (1.,0.,0.)		#GEN
		#world initialization		#GEN
				#GEN
		self.ode_world = OdeWorld()		#GEN
		self.ode_world.setGravity( 0,-9.81,0 )		#GEN
		self.ode_world.setCfm(0.000100)		#GEN
		self.ode_world.setErp(0.300000)		#GEN
		#space initialization		#GEN
				#GEN
		self.ode_space = OdeSimpleSpace()		#GEN
		#create bodies		#GEN
				#GEN
		self.mygeoms = []

		self.ode_bodies = {}		#GEN
		self.ode_mass = {}		#GEN
		self.ode_geom = {}		#GEN
		self.ode_joints = {}		#GEN
		self.ode_joint_data = {}		#GEN
		self.ode_bodies[(5,)] = OdeBody(self.ode_world)		#GEN
		#floor		#GEN
				#GEN
		self.ode_geom[(5,)] = OdePlaneGeom(self.ode_space,Vec4(-0.04,1.,0.,0))		#GEN
		self.ode_bodies[(6,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6,)] = OdeMass()		#GEN
		self.ode_mass[(6,)].setZero()		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0506,0.0355,0.0356)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6,)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.075,0.01,-0.03)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,0.0506,0.0355,0.0356)		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0.075,0.01,-0.03)		#GEN
		self.ode_mass[(6,)].add(self.ode_mass[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0506,0.0355,0.0356)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6,)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.075,0.01,0.03)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,0.0506,0.0355,0.0356)		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].translate(0.075,0.01,0.03)		#GEN
		self.ode_mass[(6,)].add(self.ode_mass[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0506,0.0355,0.0356)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6,)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-0.07,0.01,-0.059)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,0.0506,0.0355,0.0356)		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].translate(-0.07,0.01,-0.059)		#GEN
		self.ode_mass[(6,)].add(self.ode_mass[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0506,0.0355,0.0356)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6,)])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-0.07,0.01,0.059)		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,0.0506,0.0355,0.0356)		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].translate(-0.07,0.01,0.059)		#GEN
		self.ode_mass[(6,)].add(self.ode_mass[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6,)].getCenter()		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 3, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6,)].getMagnitude()>0): self.ode_bodies[(6,)].setMass(self.ode_mass[(6,)])		#GEN
		self.ode_mass[(6,)].adjust(0.6)		#GEN
		#mass = Node(AMARSI): Supervisor.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.99995,0.00999983,0],[-0.00999983,0.99995,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6,)].setPosition(0+ode_tmp[0,0],0.19+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6,)].setRotation(Mat3(0.99995,0.00999983,0,-0.00999983,0.99995,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 0, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 0, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.13,0.0024,0.17)		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-0.065,-0.0012,0)		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.13,0.0024,0.17)		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].translate(-0.065,-0.0012,0)		#GEN
		self.ode_mass[(6, 2, 0, 0)].add(self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.13,0.0024,0.13)		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.065,-0.0012,0)		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.13,0.0024,0.13)		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].translate(0.065,-0.0012,0)		#GEN
		self.ode_mass[(6, 2, 0, 0)].add(self.ode_mass[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 0, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 0, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 0, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 0, 0)].setMass(self.ode_mass[(6, 2, 0, 0)])		#GEN
		#mass = Node(FR4_BODY_PARTS): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.99995,0.00999983,0],[-0.00999983,0.99995,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 0, 0)].setPosition(0+ode_tmp[0,0],0.19+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 0, 0)].setRotation(Mat3(0.99995,0.00999983,0,-0.00999983,0.99995,0,0,0,1))		#GEN
		self.ode_joints[(6, 2, 0, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 0, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 0, 0)])		#GEN
		self.ode_joints[(6, 2, 0, 0)].set()		#GEN
		self.ode_bodies[(6, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.2106,0.035,0.0856)		#GEN
		self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-0.01,-0.019,0)		#GEN
		self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,0.2106,0.035,0.0856)		#GEN
		self.ode_mass[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].translate(-0.01,-0.019,0)		#GEN
		self.ode_mass[(6, 2, 1, 0)].add(self.ode_mass[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 1, 0)])		#GEN
		self.ode_mass[(6, 2, 1, 0)].adjust(0.6)		#GEN
		#mass = Node(MOTORS_GROUP): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.99995,0.00999983,0],[-0.00999983,0.99995,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 1, 0)].setPosition(0+ode_tmp[0,0],0.19+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 1, 0)].setRotation(Mat3(0.99995,0.00999983,0,-0.00999983,0.99995,0,0,0,1))		#GEN
		self.ode_joints[(6, 2, 1, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 1, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 1, 0)].set()		#GEN
		self.ode_bodies[(6, 2, 2, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0154,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 2, 0)])		#GEN
		self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.0154,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 2, 0)].add(self.ode_mass[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 2, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 2, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 2, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 2, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 2, 0)].setMass(self.ode_mass[(6, 2, 2, 0)])		#GEN
		#mass = Node(LEFT_FORE_HIP_SERVO): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.924909,-0.380188,0],[0.380188,0.924909,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 2, 0)].setPosition(0.0900955+ode_tmp[0,0],0.1991+ode_tmp[1,0],-0.075+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 2, 0)].setRotation(Mat3(0.924909,-0.380188,0,0.380188,0.924909,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.015,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.015,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0)].add(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 2, 0, 2, 0, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 2, 0, 2, 0, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)].setMass(self.ode_mass[(6, 2, 2, 0, 2, 0, 0)])		#GEN
		#mass = Node(LEFT_FORE_FRONT_KNEE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.922843,0.385177,0],[-0.385177,0.922843,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)].setPosition(0.111006+ode_tmp[0,0],0.14823+ode_tmp[1,0],-0.0715+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)].setRotation(Mat3(0.922843,0.385177,0,-0.385177,0.922843,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.055,0.015,0.0033)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.0275,0,0)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.055,0.015,0.0033)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0.0275,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		#mass = Node(LEFT_FORE_ANKLE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.380185,0.92491,0],[-0.92491,0.380185,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setPosition(0.0898211+ode_tmp[0,0],0.0974732+ode_tmp[1,0],-0.07483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setRotation(Mat3(0.380185,0.92491,0,-0.92491,0.380185,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeCylinderGeom(self.ode_space, 0.0075, 0.015)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setCylinder(2000,2,0.0075,0.015)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		#mass = Node(LEFT_FORE_TOE): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.380185,0.92491,0],[3.39739e-06,-1.3965e-06,-1],[-0.92491,0.380185,-3.67321e-06]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setPosition(0.110731+ode_tmp[0,0],0.0466031+ode_tmp[1,0],-0.07483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setRotation(Mat3(1, 0, 0, 0, 0, 1, 0, -1, 0))		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)],self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].set()		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)],self.ode_bodies[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setAnchor(0.0898211,0.0974732,-0.07483)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0, 2, 1, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0)].attachBodies(self.ode_bodies[(6, 2, 2, 0)],self.ode_bodies[(6, 2, 2, 0, 2, 0, 0)])		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0)].setAnchor(0.111006,0.14823,-0.0715)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 2, 0, 2, 0, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 2, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 2, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 2, 0)])		#GEN
		self.ode_joints[(6, 2, 2, 0)].setAnchor(0.0900955,0.1991,-0.075)		#GEN
		self.ode_joints[(6, 2, 2, 0)].setAxis(0,0,1)		#GEN
		self.ode_joints[(6, 2, 2, 0)].setParamFMax(200)		#GEN
		self.ode_bodies[(6, 2, 3, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0154,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 3, 0)])		#GEN
		self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.0154,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 3, 0)].add(self.ode_mass[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 3, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 3, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 3, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 3, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 3, 0)].setMass(self.ode_mass[(6, 2, 3, 0)])		#GEN
		#mass = Node(RIGHT_FORE_HIP_SERVO): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.924909,-0.380188,0],[0.380188,0.924909,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 3, 0)].setPosition(0.0900955+ode_tmp[0,0],0.1991+ode_tmp[1,0],0.075+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 3, 0)].setRotation(Mat3(0.924909,-0.380188,0,0.380188,0.924909,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.015,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.015,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0)].add(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 3, 0, 2, 0, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 3, 0, 2, 0, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)].setMass(self.ode_mass[(6, 2, 3, 0, 2, 0, 0)])		#GEN
		#mass = Node(RIGHT_FORE_FRONT_KNEE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.922843,0.385177,0],[-0.385177,0.922843,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)].setPosition(0.111006+ode_tmp[0,0],0.14823+ode_tmp[1,0],0.0715+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)].setRotation(Mat3(0.922843,0.385177,0,-0.385177,0.922843,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.055,0.015,0.0033)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.0275,0,0)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.055,0.015,0.0033)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0.0275,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		#mass = Node(RIGHT_FORE_ANKLE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.380185,0.92491,0],[-0.92491,0.380185,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setPosition(0.0898211+ode_tmp[0,0],0.0974732+ode_tmp[1,0],0.07483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setRotation(Mat3(0.380185,0.92491,0,-0.92491,0.380185,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeCylinderGeom(self.ode_space, 0.0075, 0.015)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setCylinder(2000,2,0.0075,0.015)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		#mass = Node(RIGHT_FORE_TOE): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.380185,0.92491,0],[3.39739e-06,-1.3965e-06,-1],[-0.92491,0.380185,-3.67321e-06]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setPosition(0.110731+ode_tmp[0,0],0.0466031+ode_tmp[1,0],0.07483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setRotation(Mat3(1, 0, 0, 0, 0, 1, 0, -1, 0))		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)],self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].set()		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)],self.ode_bodies[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setAnchor(0.0898211,0.0974732,0.07483)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0, 2, 1, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0)].attachBodies(self.ode_bodies[(6, 2, 3, 0)],self.ode_bodies[(6, 2, 3, 0, 2, 0, 0)])		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0)].setAnchor(0.111006,0.14823,0.0715)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 3, 0, 2, 0, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 3, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 3, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 3, 0)])		#GEN
		self.ode_joints[(6, 2, 3, 0)].setAnchor(0.0900955,0.1991,0.075)		#GEN
		self.ode_joints[(6, 2, 3, 0)].setAxis(0,0,1)		#GEN
		self.ode_joints[(6, 2, 3, 0)].setParamFMax(200)		#GEN
		self.ode_bodies[(6, 2, 4, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0154,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 4, 0)])		#GEN
		self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.0154,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 4, 0)].add(self.ode_mass[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 4, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 4, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 4, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 4, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 4, 0)].setMass(self.ode_mass[(6, 2, 4, 0)])		#GEN
		#mass = Node(LEFT_HIND_HIP_SERVO): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.958244,-0.285952,0],[0.285952,0.958244,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 4, 0)].setPosition(-0.0898955+ode_tmp[0,0],0.200899+ode_tmp[1,0],-0.095+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 4, 0)].setRotation(Mat3(0.958244,-0.285952,0,0.285952,0.958244,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.015,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.015,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0)].add(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 4, 0, 2, 0, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 4, 0, 2, 0, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)].setMass(self.ode_mass[(6, 2, 4, 0, 2, 0, 0)])		#GEN
		#mass = Node(LEFT_HIND_FRONT_KNEE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.879779,0.475384,0],[-0.475384,0.879779,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)].setPosition(-0.0741681+ode_tmp[0,0],0.148196+ode_tmp[1,0],-0.0915+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)].setRotation(Mat3(0.879779,0.475384,0,-0.475384,0.879779,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.055,0.015,0.0033)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.0275,0,0)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.055,0.015,0.0033)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0.0275,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		#mass = Node(LEFT_HIND_ANKLE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.285949,0.958245,0],[-0.958245,0.285949,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setPosition(-0.100314+ode_tmp[0,0],0.0998082+ode_tmp[1,0],-0.09483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setRotation(Mat3(0.285949,0.958245,0,-0.958245,0.285949,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeCylinderGeom(self.ode_space, 0.0075, 0.015)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setCylinder(2000,2,0.0075,0.015)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		#mass = Node(LEFT_HIND_TOE): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.285949,0.958245,0],[3.51983e-06,-1.05035e-06,-1],[-0.958245,0.285949,-3.67321e-06]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setPosition(-0.084587+ode_tmp[0,0],0.0471048+ode_tmp[1,0],-0.09483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setRotation(Mat3(1, 0, 0, 0, 0, 1, 0, -1, 0))		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)],self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].set()		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)],self.ode_bodies[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setAnchor(-0.100314,0.0998082,-0.09483)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0, 2, 1, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0)].attachBodies(self.ode_bodies[(6, 2, 4, 0)],self.ode_bodies[(6, 2, 4, 0, 2, 0, 0)])		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0)].setAnchor(-0.0741681,0.148196,-0.0915)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 4, 0, 2, 0, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 4, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 4, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 4, 0)])		#GEN
		self.ode_joints[(6, 2, 4, 0)].setAnchor(-0.0898955,0.200899,-0.095)		#GEN
		self.ode_joints[(6, 2, 4, 0)].setAxis(0,0,1)		#GEN
		self.ode_joints[(6, 2, 4, 0)].setParamFMax(200)		#GEN
		self.ode_bodies[(6, 2, 5, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.0154,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 5, 0)])		#GEN
		self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.0154,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 5, 0)].add(self.ode_mass[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 5, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 5, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 5, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 5, 0)].setMass(self.ode_mass[(6, 2, 5, 0)])		#GEN
		#mass = Node(RIGHT_HIND_HIP_SERVO): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.958244,-0.285952,0],[0.285952,0.958244,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 5, 0)].setPosition(-0.0898955+ode_tmp[0,0],0.200899+ode_tmp[1,0],0.095+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 5, 0)].setRotation(Mat3(0.958244,-0.285952,0,0.285952,0.958244,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.015,0.055,0.0033)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)])		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,-0.0275,0)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.015,0.055,0.0033)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,-0.0275,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0)].add(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 5, 0, 2, 0, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 5, 0, 2, 0, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)].setMass(self.ode_mass[(6, 2, 5, 0, 2, 0, 0)])		#GEN
		#mass = Node(RIGHT_HIND_FRONT_KNEE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.879779,0.475384,0],[-0.475384,0.879779,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)].setPosition(-0.0741681+ode_tmp[0,0],0.148196+ode_tmp[1,0],0.0915+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)].setRotation(Mat3(0.879779,0.475384,0,-0.475384,0.879779,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 0.055,0.015,0.0033)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0.0275,0,0)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1800,0.055,0.015,0.0033)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0.0275,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		#mass = Node(RIGHT_HIND_ANKLE): Servo.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.285949,0.958245,0],[-0.958245,0.285949,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setPosition(-0.100314+ode_tmp[0,0],0.0998082+ode_tmp[1,0],0.09483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setRotation(Mat3(0.285949,0.958245,0,-0.958245,0.285949,0,0,0,1))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setZero()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeCylinderGeom(self.ode_space, 0.0075, 0.015)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setCylinder(2000,2,0.0075,0.015)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].add(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getCenter()		#GEN
		self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].getMagnitude()>0): self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setMass(self.ode_mass[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		#mass = Node(RIGHT_HIND_TOE): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[0.285949,0.958245,0],[3.51983e-06,-1.05035e-06,-1],[-0.958245,0.285949,-3.67321e-06]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setPosition(-0.084587+ode_tmp[0,0],0.0471048+ode_tmp[1,0],0.09483+ode_tmp[2,0])		#GEN
		self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].setRotation(Mat3(1, 0, 0, 0, 0, 1, 0, -1, 0))		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)] = OdeFixedJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)],self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0, 2, 1, 0)].set()		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].attachBodies(self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)],self.ode_bodies[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)])		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setAnchor(-0.100314,0.0998082,0.09483)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0, 2, 1, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0)].attachBodies(self.ode_bodies[(6, 2, 5, 0)],self.ode_bodies[(6, 2, 5, 0, 2, 0, 0)])		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0)].setAnchor(-0.0741681,0.148196,0.0915)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0)].setAxis(0,0,-1)		#GEN
		self.ode_joints[(6, 2, 5, 0, 2, 0, 0)].setParamFMax(200)		#GEN
		self.ode_joints[(6, 2, 5, 0)] = OdeHingeJoint(self.ode_world)		#GEN
		self.ode_joints[(6, 2, 5, 0)].attachBodies(self.ode_bodies[(6,)],self.ode_bodies[(6, 2, 5, 0)])		#GEN
		self.ode_joints[(6, 2, 5, 0)].setAnchor(-0.0898955,0.200899,0.095)		#GEN
		self.ode_joints[(6, 2, 5, 0)].setAxis(0,0,1)		#GEN
		self.ode_joints[(6, 2, 5, 0)].setParamFMax(200)		#GEN
		self.ode_bodies[(7,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(7,)] = OdeMass()		#GEN
		self.ode_mass[(7,)].setZero()		#GEN
		self.ode_geom[(7, 2, 0, 0)] = OdeBoxGeom(self.ode_space, 0.33,0.13,0.06)		#GEN
		self.ode_geom[(7, 2, 0, 0)].setBody(self.ode_bodies[(7,)])		#GEN
		self.ode_geom[(7, 2, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(7, 2, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(7, 2, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(7, 2, 0, 0)].setBox(1000,0.33,0.13,0.06)		#GEN
		self.ode_mass[(7, 2, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(7, 2, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(7,)].add(self.ode_mass[(7, 2, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(7,)].getCenter()		#GEN
		self.ode_geom[(7, 2, 0, 0)].setOffsetPosition(self.ode_geom[(7, 2, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(7, 2, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(7, 2, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(7,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(7,)].getMagnitude()>0): self.ode_bodies[(7,)].setMass(self.ode_mass[(7,)])		#GEN
		#mass = Node(STAND): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(7,)].setPosition(10000+ode_tmp[0,0],-1.065+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(7,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(8,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(8,)] = OdeMass()		#GEN
		self.ode_mass[(8,)].setZero()		#GEN
		self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 40,0.2,40)		#GEN
		self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(8,)])		#GEN
		self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(20,-0.1,0)		#GEN
		self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,40,0.2,40)		#GEN
		self.ode_mass[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].translate(20,-0.1,0)		#GEN
		self.ode_mass[(8,)].add(self.ode_mass[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(8,)].getCenter()		#GEN
		self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(8, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(8,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(8,)].getMagnitude()>0): self.ode_bodies[(8,)].setMass(self.ode_mass[(8,)])		#GEN
		#mass = Node(RAMP): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(8,)].setPosition(10000.2+ode_tmp[0,0],-2+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(8,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(9,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(9,)] = OdeMass()		#GEN
		self.ode_mass[(9,)].setZero()		#GEN
		self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 40,0.2,40)		#GEN
		self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(9,)])		#GEN
		self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-20,-0.1,0)		#GEN
		self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,40,0.2,40)		#GEN
		self.ode_mass[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].translate(-20,-0.1,0)		#GEN
		self.ode_mass[(9,)].add(self.ode_mass[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(9,)].getCenter()		#GEN
		self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(9, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(9,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(9,)].getMagnitude()>0): self.ode_bodies[(9,)].setMass(self.ode_mass[(9,)])		#GEN
		#mass = Node(PODIUM): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(9,)].setPosition(10000.2+ode_tmp[0,0],-3+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(9,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(10,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(10,)] = OdeMass()		#GEN
		self.ode_mass[(10,)].setZero()		#GEN
		self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeSphereGeom(self.ode_space, 0.01)		#GEN
		self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(10,)])		#GEN
		self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,-4)		#GEN
		self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setSphere(1000,0.01)		#GEN
		self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setSphere(1000,0.01)		#GEN
		self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].translate(0,0,-4)		#GEN
		self.ode_mass[(10,)].add(self.ode_mass[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(10,)].getCenter()		#GEN
		self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(10, 2, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(10,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(10,)].getMagnitude()>0): self.ode_bodies[(10,)].setMass(self.ode_mass[(10,)])		#GEN
		#mass = Node(HEIGHTMAP): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(10,)].setPosition(-10000.1+ode_tmp[0,0],-4+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(10,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(11,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(11,)] = OdeMass()		#GEN
		self.ode_mass[(11,)].setZero()		#GEN
		self.ode_geom[(11, 3, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 40,0.2,40)		#GEN
		self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(11,)])		#GEN
		self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].setOffsetPosition(0,0,0)		#GEN
		self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(11, 3, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(11, 3, 0, 0, 1, 0, 0)].setBox(1000,40,0.2,40)		#GEN
		self.ode_mass[(11, 3, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(11, 3, 0, 0, 1, 0, 0)].translate(0,0,0)		#GEN
		self.ode_mass[(11,)].add(self.ode_mass[(11, 3, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(11,)].getCenter()		#GEN
		self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(11, 3, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(11,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(11,)].getMagnitude()>0): self.ode_bodies[(11,)].setMass(self.ode_mass[(11,)])		#GEN
		#mass = Node(SLOPE): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(11,)].setPosition(10000+ode_tmp[0,0],-5+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(11,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(12,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(12,)] = OdeMass()		#GEN
		self.ode_mass[(12,)].setZero()		#GEN
		self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 40,0.2,40)		#GEN
		self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(12,)])		#GEN
		self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(20,-0.1,0)		#GEN
		self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,40,0.2,40)		#GEN
		self.ode_mass[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].translate(20,-0.1,0)		#GEN
		self.ode_mass[(12,)].add(self.ode_mass[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(12,)].getCenter()		#GEN
		self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(12, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(12,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(12,)].getMagnitude()>0): self.ode_bodies[(12,)].setMass(self.ode_mass[(12,)])		#GEN
		#mass = Node(STEP_FRONT): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(12,)].setPosition(10000.2+ode_tmp[0,0],-6+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(12,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_bodies[(13,)] = OdeBody(self.ode_world)		#GEN
		ode_fix_mass = (0,0,0)		#GEN
		self.ode_mass[(13,)] = OdeMass()		#GEN
		self.ode_mass[(13,)].setZero()		#GEN
		self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeBoxGeom(self.ode_space, 40,0.2,40)		#GEN
		self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBody(self.ode_bodies[(13,)])		#GEN
		self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(-20,-0.1,0)		#GEN
		self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)] = OdeMass()		#GEN
		self.ode_mass[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setBox(1000,40,0.2,40)		#GEN
		self.ode_mass[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].rotate(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_mass[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].translate(-20,-0.1,0)		#GEN
		self.ode_mass[(13,)].add(self.ode_mass[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)])		#GEN
		ode_fix_mass = self.ode_mass[(13,)].getCenter()		#GEN
		self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].setOffsetPosition(self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[0]-ode_fix_mass[0],self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[1]-ode_fix_mass[1],self.ode_geom[(13, 3, 0, 0, 1, 0, 0, 1, 0, 0)].getOffsetPosition()[2]-ode_fix_mass[2])		#GEN
		self.ode_mass[(13,)].translate(-ode_fix_mass[0],-ode_fix_mass[1],-ode_fix_mass[2])		#GEN
		if(self.ode_mass[(13,)].getMagnitude()>0): self.ode_bodies[(13,)].setMass(self.ode_mass[(13,)])		#GEN
		#mass = Node(STEP_BACK): Solid.getMass().getMagnitude()		#GEN
		ode_tmp = dot(array([[1,0,0],[0,1,0],[0,0,1]]),array(ode_fix_mass).reshape((3,1)))		#GEN
		self.ode_bodies[(13,)].setPosition(10000.2+ode_tmp[0,0],-7+ode_tmp[1,0],0+ode_tmp[2,0])		#GEN
		self.ode_bodies[(13,)].setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		#Util		#GEN
				#GEN
		self.ode_bodies_inv = {}		#GEN
		self.ode_mass_inv = {}		#GEN
		self.ode_geom_inv = {}		#GEN
		self.ode_joints_inv = {}		#GEN
		for i in self.ode_bodies:		#GEN
			self.ode_bodies_inv[self.ode_bodies[i]] = i		#GEN
		for i in self.ode_mass:		#GEN
			self.ode_mass_inv[self.ode_mass[i]] = i		#GEN
		for geom in self.ode_geom:		#GEN
			self.ode_geom_inv[self.ode_geom[geom]] = geom		#GEN
		for joint in self.ode_joints:		#GEN
			self.ode_joints_inv[self.ode_joints[joint]] = joint		#GEN
		#total mass check
		total_mass = 0
		for m in self.ode_mass:
			if(m[0]==5):
				total_mass += self.ode_mass[m].getMagnitude()
		#print "Total mass: %f"%total_mass

		self.ode_world.initSurfaceTable(len(self.ode_bodies)) #between every pair of bodies (Webots specifies the surface properties in a physics node)
		self.ode_surface_indices = []

		for i in self.ode_bodies.values(): #defines an ordering on the bodies
			self.ode_surface_indices.append(i)

		for i in self.ode_bodies:
			body_i = self.ode_bodies[i]
			index_i = self.ode_surface_indices.index(body_i)
			node_i = self.ode_tree[i]
			if(node_i.has_property("physics") and (type(node_i["physics"][0]) is aux.NodeValue or type(node_i["physics"][0]) is aux.Use)):
				physics_i = node_i["physics"][0][0]
			else:
				physics_i = None
			if(not physics_i is None):
				bounce_i = float(physics_i.get_defproperty("bounce", [[0.5]])[0][0] )
				bounceVelocity_i = float(physics_i.get_defproperty("bounceVelocity", [[0.01]])[0][0] )
				coulombFriction_i = float(physics_i.get_defproperty("coulombFriction", [[1]])[0][0] )
				forceDependentSlip_i = float(physics_i.get_defproperty("forceDependentSlip", [[0]])[0][0] )

			for j in self.ode_bodies:
				body_j = self.ode_bodies[j]
				index_j = self.ode_surface_indices.index(body_j)
				node_j = self.ode_tree[j]
				if(node_j.has_property("physics") and (type(node_j["physics"][0]) is aux.NodeValue or type(node_j["physics"][0]) is aux.Use)):
					physics_j = node_j["physics"][0][0]
				else:
					physics_j = None
				if(not physics_j is None):
					bounce_j = float(physics_j.get_defproperty("bounce", [[0.5]])[0][0] )
					bounceVelocity_j = float(physics_j.get_defproperty("bounceVelocity", [[0.01]])[0][0] )
					coulombFriction_j = float(physics_j.get_defproperty("coulombFriction", [[1]])[0][0] )
					forceDependentSlip_j = float(physics_j.get_defproperty("forceDependentSlip", [[0]])[0][0] )
				if(not (physics_j is None or physics_i is None)):
					bounce = (bounce_i+bounce_j)/2.
					bounceVelocity = (bounceVelocity_i+bounceVelocity_j)/2.
					coulombFriction = (coulombFriction_i+coulombFriction_j)/2.
					forceDependentSlip = (forceDependentSlip_i+forceDependentSlip_j)/2.
				elif(not physics_i is None):
					bounce = bounce_i
					bounceVelocity = bounceVelocity_i
					coulombFriction = coulombFriction_i
					forceDependentSlip = forceDependentSlip_i
				elif(not physics_j is None):
					bounce = bounce_j
					bounceVelocity = bounceVelocity_j
					coulombFriction = coulombFriction_j
					forceDependentSlip = forceDependentSlip_j
				else:
					bounce = 0.5
					bounceVelocity = 0.01
					coulombFriction = 1
					forceDependentSlip = 0
				#print bounce
				#print bounceVelocity
				#print ""
				self.ode_world.setSurfaceEntry(index_i, index_j, coulombFriction, bounce, bounceVelocity, 0,  0, forceDependentSlip, 0)
				#self.ode_world.setSurfaceEntry(index_i, index_j, coulombFriction, 0, 20., 0,  0, forceDependentSlip, 0.99)
				#print "%d: %d,%d"%(len(self.ode_bodies), index_i, index_j)

		for i in self.ode_geom:
			geom = self.ode_geom[i]
			body = geom.getBody()
			#body_i = self.ode_bodies_inv[body]
			if(type(geom) is OdePlaneGeom):
				body = self.ode_bodies[i] #non-placeable geom...
			#try:
			index = self.ode_surface_indices.index(body)
			#print index
			geom.setCategoryBits(BitMask32(1<<i[0]))
			geom.setCollideBits(BitMask32(0xffffffff)^(1<<i[0]))
			self.ode_space.setSurfaceType(geom, index)
			#except ValueError:
				#pass

		self.ode_space.setAutoCollideWorld(self.ode_world)
		self.ode_contactgroup = OdeJointGroup()
		self.ode_space.setAutoCollideJointGroup(self.ode_contactgroup)

	def single_step(self):	#GEN
		self.ode_space.autoCollide()		#GEN
		self.ode_world.step(self.ode_dt)		#GEN
		self.ode_contactgroup.empty()		#GEN
		self.totaltime += self.ode_dt		#GEN

	def setFloorAngle(self,floor_angle):	#GEN
		self.ode_geom[(5,)] = OdePlaneGeom(self.ode_space,Vec4(-floor_angle,1.,0.,0.))		#GEN
	def step(self,time=0):	#GEN
		self.ode_space.autoCollide()		#GEN
		self.ode_world.quickStep(self.ode_dt)		#GEN
		self.ode_contactgroup.empty()		#GEN
		ct = self.ode_dt		#GEN
		self.totaltime += self.ode_dt		#GEN
		while ct < time:		#GEN
			self.ode_space.autoCollide()			#GEN
			self.ode_world.quickStep(self.ode_dt)			#GEN
			self.ode_contactgroup.empty()			#GEN
			ct += self.ode_dt			#GEN
			self.totaltime += self.ode_dt			#GEN
	def addBox(self, a, b, c, x, y, z):
		geom = OdeBoxGeom(self.ode_space, a,c,b)
		geom.setPosition(x,z,y)		#GEN
		geom.setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_space.setSurfaceType(geom, 17) #same as ground plane
		self.mygeoms.append(geom)
	def addSphere(self, r, x, y, z):
		geom = OdeSphereGeom(self.ode_space, r)
		geom.setPosition(x,z,y)		#GEN
		geom.setRotation(Mat3(1,0,0,0,1,0,0,0,1))		#GEN
		self.ode_space.setSurfaceType(geom, 17) #same as ground plane
		self.mygeoms.append(geom)

	def addWalk(self, xstart, xend, toplevel, floorlevel):
		self.addBox(xend-xstart, 2, toplevel-floorlevel, xend/2.+xstart/2., 0, toplevel/2.+floorlevel/2.)