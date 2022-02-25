import os

# Parametros a probar -> 
# umbral varia entre 0 y 3 
# Epocas variar entre 0, 4000
# Alpha entre 0.001 y 1
# Tolerancia se mantendra en un valor de 0.01, si el cambio de peso mayor no es ni de 0.1 no tiene sentido seguir con esa simulacion
# por que no llevara a ninguna solución óptima y sería computacionalmente costoso

umbral = 0
while umbral <= 1.5:
    if umbral == 0:
        umbral = 0.1
    else:
        umbral = umbral * 2
    epocas = 100
    while epocas <= 1000:
        epocas = epocas * 2
        alpha = 0.001
        while alpha <= 1:
            alpha = alpha * 5
            os.system(f'python perceptron.py --modo2 data/problema_real2.txt --f_out output/parameter_tuning/problema2/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')
            os.system(f'python Adaline.py --modo2 data/problema_real2.txt --f_out output/parameter_tuning/problema2/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')