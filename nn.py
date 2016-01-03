import numpy as np

import matplotlib.pyplot as plt

def predict(W1,W2,X):
    #X = np.hstack((X, np.ones_like(X)))

    W1X = W1.dot(X.T).T

    NL = np.tanh(W1X)

    #NL = np.vstack((NL, np.ones_like(NL)))

    W2NL = W2.dot(NL.T).T

    return W2NL

MAX_ITER = 10000
a = .001 #Learning rate

LAYER_SIZE = 1

X = np.array([[1,2,3,4,5,6]]).T

X_nobias = X.copy()

#X = np.hstack((X, np.ones_like(X)))

N = X.shape[0]
M = X.shape[1]

Y = np.array([[2,4,6,8,10,12]]).T

W1 = np.random.rand(LAYER_SIZE,  M)
W2 = np.random.rand(1, LAYER_SIZE)

#W1 = np.ones((M, LAYER_SIZE))

#W2 = np.ones((M, LAYER_SIZE))

for i in range(1,MAX_ITER):

    W1X = W1.dot(X.T).T

    NL = np.tanh(W1X)

    #NL = np.vstack((NL, np.ones_like(NL)))

    W2NL = W2.dot(NL.T).T

    DIFF = W2NL - Y

    SQUARED = DIFF**2

    MEAN = np.sum(SQUARED) / N

    #Backprop for W1 and W2
    mean_d_squared = 1/N * MEAN

    squared_d_diff = 2*DIFF * mean_d_squared

    diff_d_w2nl = squared_d_diff

    w2nl_d_w2 = NL.T.dot(diff_d_w2nl) #This is the gradient for w2

    w2nl_d_nl = W2.T.dot(diff_d_w2nl.T).T

    nl_d_w1x = w2nl_d_nl*((1 - NL**2))

    w1x_d_w1 = X.T.dot(nl_d_w1x).T #This is the gradient for w1

    #Perform update
    W1 = W1 - a * w1x_d_w1
    W2 = W2 - a * w2nl_d_w2

print('iteration {}'.format(i))
print('avg loss={}'.format(MEAN))
print('W1={}'.format(W1))
print('W2={}'.format(W2))
print()



X2 = np.atleast_2d(np.linspace(-50,50, num=1000)).T

Y2 = predict(W1, W2, X2)
T2 = np.tanh(X2)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(X2, Y2)
plt.plot(X2, T2)
plt.scatter(X_nobias,Y)
plt.show()
