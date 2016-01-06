from nn2 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Network = Input(2) ** Bias() ** Weight(4) ** Tanh() ** Bias() ** Weight(4) ** Tanh() ** Bias() ** Weight(1) ** Output(1)
Network.Init()

#i = np.linspace(-1,1,3)
#x, y = np.meshgrid(i, i)
#z = x + y

TrainInput = np.random.rand(500,2)*2-1
TrainOutput = TrainInput[:,0]**2 + TrainInput[:,1]**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TrainInput[:,0], TrainInput[:,1], TrainOutput, c='b')

for n in range(1, 10000):
    Network.Train(TrainInput.T, TrainOutput.T, 0.001)

TestInput = np.random.rand(500,2)*4-2
TestOutput = Network.Evaluate(TestInput.T).T

ax.scatter(TestInput[:,0], TestInput[:,1], TestOutput, c='g')

plt.show()

