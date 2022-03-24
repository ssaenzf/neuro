from multicapa import Multicapa
from leerFichero import LeerFichero
import csv
import pandas as pd

# problema_real 1, 2, 3, 4, 5, 6 Sin normalizar
# ficheros_entrada = ["data/problema_real1.txt",
#                     "data/problema_real2.txt",
#                     "data/problema_real3.txt",
#                     "data/problema_real4.txt",
#                     "data/problema_real5.txt"]
ficheros_entrada = ["data/problema_real6.txt"]

coeficientes_alpha = [0.003, 0.01, 0.03, 0.01, 0.03, 0.1, 0.3]
numero_neuronas_capa = [[], [2], [4], [8], [4, 2]]
tolerancia = 0.005
epocas = [1000, 1000]
norm = False
test_size = 0.25
fichero_scores = "data/hyperparameter_tuning_sin_normZhijie.csv"
df_cols = ['problema', 'alpha' , 'capa_oculta', 'num_epocas', 'score']
#with column names
df = pd.read_csv(fichero_scores)
df = df.drop('Unnamed: 0',axis=1)
for fichero in ficheros_entrada:
    for epoca in epocas:
        for alpha in coeficientes_alpha:
            # Se selecciona el número de capas
            for neuronas in numero_neuronas_capa:
                X_train, X_test, y_train, y_test = LeerFichero.mode1(fichero, test_size, norm=norm)
                red = Multicapa(alpha=alpha, capas_neu=neuronas, tolerancia=tolerancia, epoca=epoca)
                lista_epoca, _ = red.train(X_train, y_train, X_test, y_test)
                y_preds = red.test(X_test)
                score = round(red.score(X_test, y_test), 5)
                num_epocas = lista_epoca[-1]
                matriz = red.matriz_confusion(y_test, y_preds)
                print(f'Ejecutado: {fichero}, {alpha}, {neuronas}, {num_epocas}, {score}')
                new_row = {'problema':fichero[5:], 'alpha':alpha, 'capa_oculta': neuronas, 'num_epocas': num_epocas,'score': score}
                df = df.append(new_row, ignore_index=True)
    # Cuando se haya procesado un problema se sobreescrbira el csv
    df.to_csv(fichero_scores)
