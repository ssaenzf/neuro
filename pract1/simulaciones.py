import os

# Parametros a probar -> 
# umbral varia entre 0 y 1.5
# Epocas variar entre 0, 1000
# Alpha entre 0.001 y 1
# Tolerancia se mantendra en un valor de 0.01, si el cambio de peso mayor no es ni de 0.1 no tiene sentido seguir con esa simulacion
# por que no llevara a ninguna solución óptima y sería computacionalmente costoso

umbral = 0.1
while umbral <= 1.5:
    umbral = umbral * 2
    epocas = 10
    while epocas <= 1000:
        epocas = epocas * 2
        alpha = 0.001
        while alpha <= 0.33:
            alpha = alpha * 3
            print(f"Perceptron, umbral {umbral}, epocas {epocas}, alpha {alpha}")
            os.system(f'python perceptron.py --modo3 data/problema_real2.txt data/problema_real2_no_etiquetados.txt --f_out output/parameter_tuning/problema2/perceptron/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --epoca {epocas}')
            print(f"Adaline, umbral {umbral}, epocas {epocas}, alpha {alpha}")
            os.system(f'python Adaline.py --modo3 data/problema_real2.txt data/problema_real2_no_etiquetados.txt --f_out output/parameter_tuning/problema2/adaline/output_umbral{umbral}_epocas{epocas}_alpha{alpha}.txt --umbral {umbral} --alpha {alpha} --torelancia 0.01 --epoca {epocas}')