import timeit
import numpy as np
import time
from scipy.special import expit

def sigmoid(x):
    y = np.exp(x)
    return y/(1+y)

def sigmoid2(x):
    y = np.exp(-x)
    return 1/(1+y)

start_time = time.time()

dic = {}

num = 1
for i in range(100):
    if num in dic:
        x = dic[num]
    else:
        x = sigmoid(num)
        dic[num] = x
    print(x)


# for i in range(100):
#     print(sigmoid(num))

print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()

# for i in range(-100, 100):
#     print(sigmoid(i))

# print("--- %s seconds ---" % (time.time() - start_time))