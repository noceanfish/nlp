import numpy as np
from ann import getData, softmax


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)


def classification_rate(Y, P):
    return np.mean(Y == P)


def main():
    X, Y = getData()

    M = 5
    D = X.shape[1]
    K = len(set(Y))

    W1 = np.random.randn(D, M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = np.zeros(K)

    P_Y_given_X = forward(X, W1, b1, W2, b2)
    predicticns = np.argmax(P_Y_given_X, axis=1)

    print("classifition rate:", classification_rate(Y, predicticns))

if __name__ == "__main__":
    main()

