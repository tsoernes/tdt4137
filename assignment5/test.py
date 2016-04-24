import numpy as np
import random

np.random.seed(123)
list1 = np.random.permutation(5)
np.random.seed(123)
list2 = np.random.permutation(6)

print(list1, list2)
assert(list1==list2[:-1])