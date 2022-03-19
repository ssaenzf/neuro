from multicapa import Multicapa
from leerFichero import LeerFichero
import csv
import pandas as pd

# problema_real 1, 2, 3, 4, 5, 6 Sin normalizar
ficheros_entrada = ["data/problema_real1.txt",
                    "data/problema_real2.txt",
                    "data/problema_real3.txt",
                    "data/problema_real4.txt",
                    "data/problema_real5.txt",
                    "data/problema_real6.txt"]
coeficientes_alpha = [0.001, 0.003, 0.01, 0.03, 0.01, 0.03, 0.1, 0.3]
numero_neuronas_capa = [[], [2], [4], [8], [2, 2], [4, 2], [4, 4]]
tolerancia = 0.005
epocas = [1500, 1500, 1500]
norm = False
test_size = 0.25
fichero_scores = "data/hyperparameter_tuning_sin_norm.csv"
df_cols = ['problema', 'alpha' , 'capa_oculta', 'num_epocas', 'score']
#with column names
df = pd.DataFrame(columns=df_cols)
for fichero in ficheros_entrada:
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
                # TODO: csv con formato: problema, alpha, capa_oculta, num_epocas, score
                new_row = {'problema':fichero[5:], 'alpha':alpha, 'configuracion_capas_neurona': neuronas, 'score': score}
                #append row to the dataframe
                df = df.append(new_row, ignore_index=True)
                print(f'Ejecutado: {fichero}, {alpha}, {neuronas}, {num_epocas}, {score}')
    # Cuando se haya procesado un problema se sobreescrbira el csv
    df.to_csv(fichero_scores)

# problema_real 4, 6 Con normalizar
ficheros_entrada = ["data/problema_real4.txt",
                    "data/problema_real6.txt"]
numero_neuronas_capa = [[], [2], [4], [8], [2, 2], [4, 2], [4, 4]]
tolerancia = 0.005
coeficientes_alpha = [0.001, 0.003, 0.01, 0.03, 0.01, 0.03, 0.1, 0.3]
epocas = [1500, 1500, 1500]
norm = True
test_size = 0.25
fichero_scores = "data/hyperparameter_tuning_con_norm.csv"
df = pd.DataFrame(columns=df_cols)
for fichero in ficheros_entrada:
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
                new_row = {'problema':fichero[5:], 'alpha':alpha, 'configuracion_capas_neurona': neuronas, 'score': score}
                #append row to the dataframe
                df = df.append(new_row, ignore_index=True)
                # TODO: csv con formato: problema, alpha, capa_oculta, num_epocas, score, matriz_confusion
                print(f'Ejecutado: {fichero}, {alpha}, {neuronas}, {num_epocas}, {score}')
                # f.writelines([f"Validacion con red neuronal con 1 capa oculta con {neuronas} neuronas, alpha {alpha}, epocas {epoca}", 
                #               f"Porcentaje de aciertos: {red.score(X_test, y_test)}%\n"])
    # Cuando se haya procesado un problema se sobreescrbira el csv
    df.to_csv(fichero_scores)


# problema_real 2
# numero_capas = [1, 2, 3, 4, 5, 6, 7, 8]
# numero_neuronas_capa = [1, 3, 8, 15, 30, 60]
# tolerancia = 0.01
# coeficientes_alpha = [0.001, 0.005, 0.01, 0.03, 0.06, 0.01, 0.015, 0.02, 0.03, 0.04]
# epocas = [100, 600, 2000]
# norm = True
# test_size = 0.25
# fichero_scores = "data/Scores_problema2.txt"
# f = open(fichero_scores, "w")
# f.writelines(['Problema real 2'])
# for epoca in epocas:
#     for alpha in coeficientes_alpha:
#         # Se selecciona el número de capas
#         for neuronas in numero_neuronas_capa:
#             X_train, X_test, y_train, y_test = LeerFichero.mode3('data/problema_real2.txt', 'data/problema_real2_no_etiquetados.txt', norm=norm)
#             red = Multicapa(alpha=alpha, capas_neu=[neuronas], tolerancia=tolerancia, epoca=epoca)
#             red.train(X_train, y_train, X_test, y_test)
#             y_preds = red.test(X_test)
#             f.writelines([f"Validacion con red neuronal con 1 capa oculta con {neuronas} neuronas, alpha {alpha}, epocas {epoca}", 
#                           f"Porcentaje de aciertos: {red.score(X_test, y_test)}%\n"])
# f.close()
    