import numpy as np

from .DynamicalSystem import DynamicalSystem
from .PlanarForwardKinematics import PlanarForwardKinematics

class Pendulum(DynamicalSystem, PlanarForwardKinematics):

    def __init__(self):
        PlanarForwardKinematics.__init__(self, 1)
        DynamicalSystem.__init__(self, 1)

        self.maxTorque = np.array([30])
        self.noiseState = 0
        self.stateMinRange = np.array([-np.pi, -20])
        self.stateMaxRange = np.array([ np.pi,  20])

        self.stateNames = ['theta', 'thetaDot']
        self._lengths = 0.5
        self._masses = 5
        self._updateInertias()
        self.g = 9.81
        self.sim_dt = 1e-4
        self.friction = 1

    @property
    def lengths(self):
        return self._lengths

    @lengths.setter
    def lengths(self, lengths):
        assert(lengths > 0)
        self._lengths = lengths
        self._updateInertias()

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, masses):
        assert(masses > 0)
        self._masses = masses
        self._updateInertias()

    @property
    def inertias(self):
        return self._inertias

    def _updateInertias(self):
        self._inertias = self._masses * self._lengths**2 / 3

    def transitionFunction(self, states, actions):
        actions = np.maximum(-self.maxTorque, np.minimum(actions, self.maxTorque))
        actionNoise = actions + self.getControlNoise(states, actions)
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.lengths * self.masses / self.inertias
        for i in range(0, int(nSteps)):
            velNew = states[:, 1:2] + self.sim_dt * (c * np.sin(states[:, 0:1])
                                                     + actionNoise / self.inertias
                                                     - states[:, 1:2] * self.friction )
            states = np.concatenate((states[:, 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states
