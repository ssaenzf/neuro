import os

# Parametros a probar -> 
# umbral varia entre 0 y 1.5
# Epocas variar entre 0, 1000
# Alpha entre 0.001 y 1
# Tolerancia se mantendra en un valor de 0.01, si el cambio de peso mayor no es ni de 0.1 no tiene sentido seguir con esa simulacion
# por que no llevara a ninguna solución óptima y sería computacionalmente costoso

# umbral = 0.1
# while umbral <= 1:
#     umbral += 0.1
#     epocas = 100
#     while epocas <= 1000:
#         epocas += 100
#         alpha = 0.01
#         while alpha <= 1:
#             alpha = alpha * 3
#             print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#             os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')
#             print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#             os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')


umbral = 0.01
epocas = 500
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

umbral = 0.1
epocas = 200
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')


print("============================================")
umbral = 0.01
epocas = 500
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

umbral = 0.1
epocas = 200
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')

print("============================================")
umbral = 0.01
epocas = 500
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

umbral = 0.1
epocas = 200
alpha = 0.01
while alpha <= 1:
    alpha = alpha * 3
    print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')