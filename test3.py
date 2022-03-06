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
for num in [1, 1.1, 1.11, 1.111, 1.1111]:
    if num in dic:
        x = dic[num]
        print('a')
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