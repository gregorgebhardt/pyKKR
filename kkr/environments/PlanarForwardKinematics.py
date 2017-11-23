import numpy as np

class PlanarForwardKinematics():

    def __init__(self, dimensions):
        self._lengths = np.ones((1, dimensions))
        self.numJoints = dimensions
        self.offSet = np.asarray([0, 0])

    @property
    def lengths(self):
        return self._lengths

    @lengths.setter
    def lengths(self, lengths):
        assert(lengths > 0)
        self._lengths = lengths

    def getForwardKinematics(self, theta, numLink=None):
        if numLink == None:
            numLink = self.numJoints

        y = np.zeros(np.shape(theta)[0], 2)
        for i in range(0, numLink):
            y += self.anglesToLine(theta, i)
        return y + self.offSet

    def getTaskSpaceVelocity(self, jointPositions, jointVelocities):
        taskSpaceVelocity = np.zeros(np.shape(jointVelocities)[0], 2)
        for i in range(0, jointVelocities):
            J, _ = self.getJacobian(jointPositions[i, :])
            taskSpaceVelocity[i,:] = np.matmul(J, np.transpose(jointVelocities[i, :]))

    def getJacobian(self, theta, numLink=None):
        if numLink == None:
            numLink = self.numJoints

        si = self.getForwardKinematics(theta, numLink)
        J = np.zeros(2, self.numJoints)

        for j in range(0, numLink-1):
            pj = [0, 0]
            for i in range(0, j):
                pj += self.anglesToLine(theta, i)
            pj = -(si - pj)
            J[0:1, j+1] = np.asarray([-pj(1), pj(0)])
        return J, si

    def anglesToLine(self, theta, i):
        accumulatedTheta = np.sum(theta[0:i])
        return np.asarray([np.sin(accumulatedTheta), np.cos(accumulatedTheta)]) * self.lengths[i]
