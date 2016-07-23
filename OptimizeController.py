import theano


outputs, updates = theano.scan(
    fn=lambda a,b,c: self.physics.step_from_this_state((a,b,c), dt=DT, motor_signals=[-cos(ph),cos(ph),-cos(ph),cos(ph)]+
                                                                                               [sin(ph),0,0,-sin(ph),0,0,sin(ph),0,0,-sin(ph),0,0]),
    outputs_info=self.physics.getState(),
    n_steps=2
)