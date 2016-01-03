import numpy as np

import matplotlib.pyplot as plt

def predict(W1,W2,X):
    if W1_BIAS:
        X = np.hstack((X, np.ones_like(X)))

    W1X = W1.dot(X.T).T

    NL = np.tanh(W1X)

    if W2_BIAS:
        NL_BIAS = np.hstack((NL, np.ones_like(NL)))
        W2NL = W2.dot(NL_BIAS.T).T
    else:
        W2NL = W2.dot(NL.T).T

    return W2NL

MAX_ITER = 10000
a = .01 #Learning rate

W1_BIAS = True
W2_BIAS = False

S = 1

X = np.atleast_2d(np.arange(-10,10, dtype=np.float)).T

Y = (X - 10)*2

#Normalize input

X_MEAN = np.mean(X)

X -= X_MEAN

X_STD = np.std(X)

X /= X_STD

#Normalize target
Y_MEAN = np.mean(Y)

Y -= Y_MEAN

Y_STD = np.std(Y)

Y /= Y_STD

X_nobias = X.copy()

if W1_BIAS:
    X = np.hstack((X, np.ones_like(X)))

N = X.shape[0]
M = X.shape[1]


W1 = np.random.rand(S,  M)

if W2_BIAS:
    W2 = np.random.rand(1, S+1) #+1 is for the bias
else:
    W2 = np.random.rand(1, S)

#W1 = np.ones((M, LAYER_SIZE))

#W2 = np.ones((M, LAYER_SIZE))

ERR_MIN = np.inf
W1_BEST = W1
W2_BEST = W2

for i in range(1,MAX_ITER):

    W1X = W1.dot(X.T).T

    NL = np.tanh(W1X)

    if W2_BIAS:
        NL_BIAS = np.hstack((NL, np.ones_like(NL)))
        W2NL = W2.dot(NL_BIAS.T).T
    else:
        W2NL = W2.dot(NL.T).T

    DIFF = W2NL - Y

    SQUARED = DIFF**2

    MEAN = np.sum(SQUARED) / N

    #Backprop for W1 and W2
    mean_d_squared = 1/N * MEAN

    squared_d_diff = 2*DIFF * mean_d_squared

    diff_d_w2nl = squared_d_diff

    if W2_BIAS:
        w2nl_d_w2 = NL_BIAS.T.dot(diff_d_w2nl).T #This is the gradient for w2
        w2nl_d_nl = W2.T.dot(diff_d_w2nl.T).T
        #Keep only S dimensions, ignore gradient for constant input 1
        w2nl_d_nl = w2nl_d_nl[:,:S]
    else:
        w2nl_d_w2 = NL.T.dot(diff_d_w2nl).T #This is the gradient for w2
        w2nl_d_nl = W2.T.dot(diff_d_w2nl.T).T

    nl_d_w1x = w2nl_d_nl*(1 - NL**2)

    w1x_d_w1 = X.T.dot(nl_d_w1x).T #This is the gradient for w1

    #Perform update
    W1 = W1 - a * w1x_d_w1
    W2 = W2 - a * w2nl_d_w2

    if MEAN < ERR_MIN:
        W1_BEST = W1
        W2_BEST = W2
        ERR_MIN = MEAN

print('iteration {}'.format(i))
print('avg loss={}'.format(MEAN))
print('W1={}'.format(W1))
print('W2={}'.format(W2))
print()

W1 = W1_BEST
W2 = W2_BEST

print('avg loss={}'.format(ERR_MIN))
print('W1={}'.format(W1_BEST))
print('W2={}'.format(W2_BEST))
print()


X2 = np.atleast_2d(np.linspace(-2,2, num=1000)).T
Y2 = predict(W1, W2, X2)

X2 = X2 * X_STD + X_MEAN
Y2 = Y2 * Y_STD + Y_MEAN

T2 = np.tanh(X2)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(X2, Y2)
plt.plot(X2, T2)
plt.scatter(X_nobias*X_STD + X_MEAN,Y*Y_STD + Y_MEAN)
plt.show()
