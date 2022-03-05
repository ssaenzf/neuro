import os
import matplotlib.pyplot as plt
#os.system(f'python perceptron.py --modo2 data/problema_real1.txt --f_out output/prob_real1_perceptron.txt --umbral 0.1 --alpha 0.125 --epoca 800')
#os.system(f'python Adaline.py --modo2 data/problema_real1.txt --f_out output/prob_real1_adaline.txt --umbral 0.1 --alpha 0.025 --torelancia 0.01 --epoca 200')
X = []
Y = []
# Umbral, epocas y coeficiente
# Perceptron umbral, epocas 200 y alpha 0.005
"""
X.append(10)
Y.append(96.40287769784173)

X.append(20)
Y.append(95.68345323741008)

X.append(40)
Y.append(97.12230215827337)

X.append(60)
Y.append(94.96402877697841)

X.append(80)
Y.append(95.68345323741008)

X.append(120)
Y.append(96.40287769784173)    

X.append(200)
Y.append(94.96402877697841)

X.append(300)
Y.append(93.5251798561151) 

X.append(500)
Y.append(96.40287769784173) 

X.append(700)
Y.append(95.68345323741008) 

plt.plot(X, Y)
plt.show()

"""
for epoca in [10, 20, 40, 60, 80, 120, 200, 300, 500, 700]:
    os.system(f'python3 perceptron.py --modo1 data/problema_real1.txt 0.2 --umbral 0.25 --alpha 0.125 --epoca {epoca}')
