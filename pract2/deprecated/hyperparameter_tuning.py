ficheros_entrada = ["data/problema_real1.txt", "data/problema_real2.txt", "data/problema_real2_no_etiquetados.txt", "data/problema_real3.txt", "data/problema_real5.txt"]
numero_capas = [1, 2, 3, 4, 5, 6, 7, 8]
numero_neuronas_capa = [1, 3, 8, 15, 30, 60]
tolerancia = 0.01
coeficientes_alpha = [0.001, 0.005, 0.01, 0.03, 0.06, 0.01, 0.015, 0.02, 0.03, 0.04]
epocas = [100, 600, 2000]

"""
# Recibe un diccionario con configuracion de capas y devuelve la siguiente configuración y si esta existe o sobrepasa el número de capas
def siguiente_configuracion_capas(dict_configuracion_capas, num_capas):
    existe = False
    if dict_configuracion_capas[i] == 5 and i==0:
        existe = False
    elif dict_configuracion_capas[i] == 5:
        dict_configuracion_capas[i] = 0
        existe, dict_configuracion_capas = siguiente_configuracion_capas(dict_configuracion_capas, num_capas)
    else:
        dict_configuracion_capas[i] = dict_configuracion_capas[i] + 1
        existe = True
    return existe, dict_configuracion_capas
"""
# Se comienza iterando por los diferentes problemas
for fichero in ficheros_entrada:
    # Se continúa iterando seleccionando el coeficiente alpha y el número de epocas
    for epoca in epocas:
        for alpha in coeficientes_alpha:
            # Se selecciona el número de capas
            for neuronas in numero_neuronas_capa:
                pass


            """
            for capas in numero_capas:
                # Inicialización configuración capas, aqui van todas las posibilidades para el conjunto de capas
                dict_configuracion_capas = {}
                for i in range (capas):
                    dict_configuracion_capas[i] = 0
                existe_siguiente = True
                t = 0
                while(existe_siguiente):
                    if t != 0: # En la primera iteraccion no se busca siguiente porque ya se tiene la primera configuracion disponible para simulacion
                        existe_siguiente, dict_configuracion_capas = siguiente_configuracion_capas(dict_configuracion_capas, capas)
                    t = t + 1
                    if existe_siguiente:           
                    # Simulación con conjunto de parametros 
                    
                # Tenemos el número de capas
            """

"""
X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1], norm=norm)
red = Multicapa(alpha=alpha, capas_neu=capas_neu, tolerancia=torelancia, epoca=epoca)
red.train(X_train, y_train, X_test, y_test)
y_preds = red.test(X_test)
print("Porcentaje de aciertos: {}%\n".format(red.score(X_test, y_test)))
"""