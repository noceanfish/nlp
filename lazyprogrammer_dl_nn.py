import numpy as np

w = np.array([[0.5, 0.1, -0.3], [0.7, -0.3, 0.2]])
b = np.array([0.4, 0.1, 0])

x = np.array([[0, 3.5], [1, 2], [1, 0.5]])

print(w.T.shape)
print(b.shape)
print(np.transpose(b))
print(x)

y1 = w.T.dot(x.T) + b.T
y = np.tanh(x.dot(w) + b.T)
print(y)
print(y1)
