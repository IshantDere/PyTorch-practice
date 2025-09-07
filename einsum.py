import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.einsum("i,i->", a, b) # (1*4)+()
print(result)