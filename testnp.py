import numpy as np

c = np.array([[1, 2, 0, 10/3], [0, 0, 1, -1/3], [1, 0, 0, 0], [0, 1, 0, 0]])
b = np.array([4, 4, 1, 1])

x = np.linalg.solve(c, b)

print(x)