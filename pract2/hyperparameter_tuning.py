from multicapa import Multicapa
from leerFichero import LeerFichero

# problema_real 1, 3 , 5
ficheros_entrada = ["data/problema_real1.txt", "data/problema_real3.txt", "data/problema_real5.txt"]
numero_capas = [1, 2, 3, 4, 5, 6, 7, 8]
numero_neuronas_capa = [1, 3, 8, 15, 30, 60]
tolerancia = 0.01
coeficientes_alpha = [0.001, 0.005, 0.01, 0.03, 0.06, 0.01, 0.015, 0.02, 0.03, 0.04]
epocas = [100, 600, 2000]
norm = True
test_size = 0.25
fichero_scores = "data/Scores_problemas_1_3_5.txt"
f = open(fichero_scores, "w")
for fichero in ficheros_entrada:
    f.writelines([fichero])
    for epoca in epocas:
        for alpha in coeficientes_alpha:
            # Se selecciona el número de capas
            for neuronas in numero_neuronas_capa:
                X_train, X_test, y_train, y_test = LeerFichero.mode1(fichero, test_size, norm=norm)
                red = Multicapa(alpha=alpha, capas_neu=[neuronas], tolerancia=tolerancia, epoca=epoca)
                red.train(X_train, y_train, X_test, y_test)
                y_preds = red.test(X_test)
                f.writelines([f"Validacion con red neuronal con 1 capa oculta con {neuronas} neuronas, alpha {alpha}, epocas {epoca}", 
                              f"Porcentaje de aciertos: {red.score(X_test, y_test)}%\n"])
f.close()

# problema_real 2
numero_capas = [1, 2, 3, 4, 5, 6, 7, 8]
numero_neuronas_capa = [1, 3, 8, 15, 30, 60]
tolerancia = 0.01
coeficientes_alpha = [0.001, 0.005, 0.01, 0.03, 0.06, 0.01, 0.015, 0.02, 0.03, 0.04]
epocas = [100, 600, 2000]
norm = True
test_size = 0.25
fichero_scores = "data/Scores_problema2.txt"
f = open(fichero_scores, "w")
f.writelines(['Problema real 2'])
for epoca in epocas:
    for alpha in coeficientes_alpha:
        # Se selecciona el número de capas
        for neuronas in numero_neuronas_capa:
            X_train, X_test, y_train, y_test = LeerFichero.mode3('data/problema_real2.txt', 'data/problema_real2_no_etiquetados.txt', norm=norm)
            red = Multicapa(alpha=alpha, capas_neu=[neuronas], tolerancia=tolerancia, epoca=epoca)
            red.train(X_train, y_train, X_test, y_test)
            y_preds = red.test(X_test)
            f.writelines([f"Validacion con red neuronal con 1 capa oculta con {neuronas} neuronas, alpha {alpha}, epocas {epoca}", 
                          f"Porcentaje de aciertos: {red.score(X_test, y_test)}%\n"])
f.close()
    