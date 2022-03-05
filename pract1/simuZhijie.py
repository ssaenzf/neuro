import os


# epocas = 500
# alpha = 0.1
# for umbral in [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5]:
#     print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#     os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

# epocas = 500
# alpha = 0.1
# for umbral in [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5]:
#     print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#     os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')

# print("============================================")
# umbral = 0.1
# epocas = 500
# for alpha in [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3]:
#     print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#     os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

# umbral = 0.05
# epocas = 500
# for alpha in [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3]:
#     print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
#     os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')


print("============================================")
umbral = 0.1
alpha = 0.2
for epocas in [100, 200, 300, 400, 500, 600, 700]:
    print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python perceptron.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')

umbral = 0.05
alpha = 0.1
for epocas in [100, 200, 300, 400, 500, 600, 700]:
    print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
    os.system(f'python adaline.py --modo1 data/problema_real1.txt 0.25 --f_out output/parameter_tuning/problema1/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')