import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    expa = np.exp(x)
    print('with:', expa / expa.sum(axis=1, keepdims=True))
    print('without:', expa / expa.sum())
    return expa / expa.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def forward(x, w1, b1, w2, b2):
    t = x.dot(w1) + b1
    z = sigmoid(t)

    return softmax(z.dot(w2) + b2)


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
