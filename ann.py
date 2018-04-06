import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getData():
    df = pd.read_csv('./ecommerce_data.csv')
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z
    return X2, Y


def get_binary_data():
    X, Y = getData()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


def softmax(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def forward(x, w1, b1, w2, b2):
    t = x.dot(w1) + b1
    z = sigmoid(t)
    s = z.dot(w2) + b2
    return softmax(s)


def classification_rate(Y, P):
    n_corrext = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_corrext += 1
    return float(n_corrext) / n_total


def main():
    Nclass = 500
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

    D = 2
    M = 3
    K = 3

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)

    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    P_Y_given_X = forward(X, W1, b1, W2, b2)
    p = np.argmax(P_Y_given_X, axis=1)

    assert(len(p) == len(Y))
    print("rate for random:", classification_rate(Y, p))


if __name__ == "__main__":
    main()
