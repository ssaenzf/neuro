from multicapa import Multicapa
from leerFichero import LeerFichero
import csv

# TODO: https://www.pythontutorial.net/python-basics/python-write-csv-file/
header = ['problema', 'alpha', 'capa_oculta', 'num_epocas', 'score', 'matriz_confusion']

# problema_real 1, 2, 3, 4, 5, 6 Sin normalizar
ficheros_entrada = ["data/problema_real1.txt"]
coeficientes_alpha = [0.01]
numero_neuronas_capa = [[], [2], [2, 2]]
tolerancia = 0.005
epocas = [1000, 1000, 1000]
norm = False
test_size = 0.25
fichero_scores = "data/hyperparameter_tuning_sin_norm.csv"
f = open(fichero_scores, "w")
for fichero in ficheros_entrada:
    f.writelines([fichero])
    for epoca in epocas:
        for alpha in coeficientes_alpha:
            # Se selecciona el número de capas
            for neuronas in numero_neuronas_capa:
                X_train, X_test, y_train, y_test = LeerFichero.mode1(fichero, test_size, norm=norm)
                red = Multicapa(alpha=alpha, capas_neu=neuronas, tolerancia=tolerancia, epoca=epoca)
                lista_epoca, _ = red.train(X_train, y_train, X_test, y_test)
                y_preds = red.test(X_test)
                score = red.score(X_test, y_test)
                num_epocas = lista_epoca[-1]
                matriz = red.matriz_confusion(y_test, y_preds)
                # TODO: csv con formato: problema, alpha, capa_oculta, num_epocas, score, matriz_confusion
                print(f'Ejecutado: {fichero}, {alpha}, {neuronas}, {num_epocas}, {score}')
                f.writelines([f"Validacion con red neuronal con 1 capa oculta con {neuronas} neuronas, alpha {alpha}, epocas {epoca}", 
                              f"Porcentaje de aciertos: {red.score(X_test, y_test)}%\n"])
f.close()