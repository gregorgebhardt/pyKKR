import numpy as np

class DynamicalSystem():

    def __init__(self, actionDim, stateDim = None):
        if (not stateDim):
            stateDim = actionDim * 2
            
        self.actionDim = actionDim
        self.stateDim = stateDim

        self.noise_std = 1
        self.noiseMode = 0
        self.registerControlNoise = False

        self.dt = 0.05


    def getControlNoise(self, states, actions):
        std = self.getControlNoiseStd(states, actions)
        return np.random.normal(loc=0.0, scale=std, size=np.shape(actions))

    def getControlNoiseStd(self, states, actions):
        if self.noiseMode == 0:
            return self.noise_std * np.ones(np.shape(actions)) / np.sqrt(self.dt)
        elif self.noiseMode == 1:
            return self.noise_std * np.abs(actions) / np.sqrt(self.dt)