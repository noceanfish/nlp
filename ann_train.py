import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from ann import getData, softmax


def y2indicator(y, K):
    N = len(y)
    t = np.zeros((N, K))
    for i in range(N):
        t[i, y[i]] = 1
    return t


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z


def predict(py_given_X):
    return np.argmax(py_given_X, axis=1)


def classification_rate(Y, P):
    return np.mean(Y == P)


def cross_entropy(T, py):
    return -np.mean(T * np.log(py))


def main():
    X, Y = getData()
    X, Y = shuffle(X, Y)
    Y = Y.astype(np.int32)

    M = 5
    D = X.shape[1]
    K = len(set(Y))

    x_train = X[: -100]
    y_train = Y[: -100]
    y_train_ind = y2indicator(y_train, K)

    x_test = X[-100:]
    y_test = Y[-100:]
    y_test_ind = y2indicator(y_test, K)

    W1 = np.random.randn(D, M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = np.zeros(K)

    train_costs = []
    test_costs = []
    learning_rate = 0.001

    for epoch in range(10000):
        pYtrain, Ztrain = forward(x_train, W1, b1, W2, b2)
        pYtest, Ztest = forward(x_test, W1, b1, W2, b2)

        ctrain = cross_entropy(y_train_ind, pYtrain)
        ctest = cross_entropy(y_test_ind, pYtest)
        train_costs.append(ctrain)
        test_costs.append(ctest)

        W2 -= learning_rate * Ztrain.T.dot(pYtrain - y_train_ind)
        b2 -= learning_rate * (pYtrain - y_train_ind).sum()
        dZ = (pYtrain - y_train_ind).dot(W2.T)*(1 - Ztrain * Ztrain)
        W1 -= learning_rate * x_train.T.dot(dZ)
        b1 -= learning_rate * dZ.sum(axis=0)
        if epoch % 100 == 0:
            print("epoch:", epoch, "train_cost:", ctrain, "test cost:", ctest)

    print("finanl train rate:", classification_rate(y_train, predict(pYtrain)))
    print("finanl test rate:", classification_rate(y_test, predict(pYtest)))

    plt.legend([plt.plot(train_costs, label='train_cost'), plt.plot(test_costs, label='test_cost')])
    plt.show()
    print()


if __name__ == "__main__":
    main()