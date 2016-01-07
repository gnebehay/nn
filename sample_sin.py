from nn2 import *
import matplotlib.pyplot as plt

Network = Input(1) ** Bias() ** Weight(4) ** Tanh() ** Bias() ** Weight(1) ** Output(1)
Network.Init()

X = np.arange(-1, 1, 0.05, dtype=np.float)
Y = np.sin(X*4)
X2 = np.arange(-5, 5, 0.05, dtype=np.float)

fig = plt.figure()
plt.scatter(X, Y)

T2 = Network.Evaluate(X2)[0]
plt.plot(X2, T2, alpha = 0.1)

for i in np.arange(0.2, 1, 0.1):
    for n in range(1, 1000):
        Network.Train(X, Y, 0.001)

    T2 = Network.Evaluate(X2)[0]
    plt.plot(X2, T2, 'g', alpha = i)

plt.plot(X2, T2, 'r')

plt.show()
