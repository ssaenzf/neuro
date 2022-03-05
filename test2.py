import timeit
import numpy as np
import time
from scipy.special import expit

x = [1,2,3,4,5,6]

x_retro = x[-2::-1]
for i in range(len(x_retro)):
    print(x_retro[i])