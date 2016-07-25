import math
import theano
from TheanoPhysicsSystem import TheanoRigid3DBodyEngine
# step 1: load the physics model
engine = TheanoRigid3DBodyEngine()
engine.load_robot_model("robotmodel/predator.json")
engine.compile()


# step 2: build the model, controller and engine for simulation
DT = 0.001
total_time = 10

def build_controller(sensor_values):
    pass

def build_objectives(states):
    pass

def build_model():

    def control_loop(positions, velocities, rot_matrices):
        sensor_values = engine.getSensorValues(state=(positions, velocities, rot_matrices))
        motor_signals = build_controller(sensor_values)
        return engine.step_from_this_state(state=(positions, velocities, rot_matrices), dt=DT, motor_signals=motor_signals)


    outputs, updates = theano.scan(
        fn=lambda a,b,c: control_loop((a,b,c)),
        outputs_info=engine.getInitialState(),
        n_steps=int(math.ceil(total_time/DT))
    )
