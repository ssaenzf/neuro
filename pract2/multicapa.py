from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from leerFichero import LeerFichero
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class Multicapa():

    # Funcion para la construccion de la red
    def __init__(self, alpha=0.3, capas_neu=[], tolerancia=0.0, epoca=100):
        self.red = RedNeuronal()
        self.alpha = alpha
        self.capas_neu = capas_neu
        self.tolerancia = tolerancia
        self.epoca = epoca

    def make_red(self, n_atributos, n_clases):
        # Capa entrada
        capa_entrada = Capa(name='Capa Entrada')
        # Neurona correspondiente al bias
        capa_entrada.aniadir(Neurona(tipo=Tipo.SESGO, name=f'X0'))
        for i in range(n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            capa_entrada.aniadir(Neurona(tipo=Tipo.DIRECTA, name=f'X{i+1}'))
        
        # Capas ocultas
        capas_ocultas = []
        neu_name = 'Z'
        n_capa = 1
        for n_neu in self.capas_neu:
            capa_oculta = Capa(name=f'Capa Oculta {n_capa}')
            # Neurona correspondiente al bias
            capa_oculta.aniadir(Neurona(tipo=Tipo.SESGO, name=f'{neu_name}0'))
            for i in range(n_neu):
                capa_oculta.aniadir(Neurona(tipo=Tipo.SIGMOIDE, name=f'{neu_name}{i+1}'))
            capas_ocultas.append(capa_oculta)

            neu_name += 'Z'
            n_capa += 1

        # Capa salida
        capa_salida = Capa(name='Capa Salida')
        for i in range(n_clases):
            capa_salida.aniadir(Neurona(tipo=Tipo.SIGMOIDE, name=f'Y{i+1}'))
        
        # Se añaden las capas a la red
        self.red.aniadir(capa_entrada)
        for capa_oculta in capas_ocultas:
            self.red.aniadir(capa_oculta)
        self.red.aniadir(capa_salida)

        # Conexiones entre capas. -1 porque el ultimo no tiene conexiones 
        for i in range(len(self.red.capas) - 2):
            # Se pone flag bias porque no queremos que cada i se conecta con bias de capa j
            self.red.capas[i].conectar(self.red.capas[i+1], -0.5 , 0.5, bias=True)
        self.red.capas[-2].conectar(self.red.capas[-1], -0.5 , 0.5)

    # Funcion para la realizacion del entrenamiento por el red
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Se crea la red del red
        self.make_red(X_train.shape[1], y_train.shape[1])
        # print(self.red)

        # Paso 0, inicial todos los pesos y sesgo
        self.red.inicializar()
        epoca = 0
        list_epoca = []
        list_ecm = []
        mejor_score = 0
        mejor_score_repetido = 0

        # Paso 1, mientras que haya actualizacion de peso, se ejecutra paso 2-9
        parar = False
        while not parar and epoca < self.epoca:
            # Se pone parar a True suponiendo que no va a actualizar durante la epoca
            parar = True

            # Uso para el calculo del error cuadratico medio en cada epoca
            epoca += 1
            error_cuad_med = 0

            # Paso 2, para cada par de entrenamiento, ejecutar paso 3-8
            for m in range(len(y_train)):
                record_x = X_train[m]
                record_y = y_train[m]

                # Paso 3, establecer las activaciones a las neuronas de entrada, a excepcion del bias 
                for i in range(len(self.red.capas[0].neuronas) - 1):
                    self.red.capas[0].neuronas[i+1].inicializar(record_x[i])
                
                # Paso 4, calcular la respuesta de las neuronas de las capas ocultas
                for capa in self.red.capas[:-1]:
                    capa.disparar()
                    capa.inicializar()
                    capa.propagar()

                # Paso 5, calcular la respuesta de las neuronas de la capa de salida
                # No se inicializa esta capa del momento, ya que necesita utilizar y_in para ajustar los pesos
                self.red.capas[-1].disparar()
                # print(self.red.capas[-1])

                # Prepara lista para almacenar los cambios de peso
                cambios_pesos = []

                # Paso 6, cada neurona de salida Yk recibe un patron objetivo que corresponde al patron de entrada
                sigmas_k = []
                deltas_w = []
                capa = self.red.capas[-1]
                for k in range(len(capa.neuronas)):
                    y_k = capa.neuronas[k].valor_salida
                    error = record_y[k] - y_k
                    error_cuad_med += error**2
                    sigma_k = error * 1/2 *  (1 + y_k) * (1 - y_k)
                    sigmas_k.append(sigma_k)

                    # Primera Capa oculta contanto desde atras
                    deltas_w_jk = []
                    # print(self.red.capas[-2])
                    for neurona in self.red.capas[-2].neuronas:
                        # delta_w_jk es el peso de la neurona de entrada j
                        # para neurona[0] que es bias, la salida es 1
                        delta_w_jk = self.alpha * sigma_k * neurona.valor_salida
                        deltas_w_jk.append(delta_w_jk)
                    deltas_w.append(deltas_w_jk)
                cambios_pesos.append(deltas_w)

                # Paso 7 y 8, retropropagacion hacia atras de X capas hasta las conexiones de capa de entrada
                for i in range(len(self.red.capas) - 2, 0, -1):
                    sigmas_j = []
                    deltas_v = []
                    capa = self.red.capas[i]
                    for j in range(1, len(capa.neuronas)):
                        sigma_in_j = 0
                        for k in range(len(sigmas_k)):
                            sigma_in_j += sigmas_k[k] * capa.neuronas[j].conexiones[k].peso

                        z_j = capa.neuronas[j].valor_salida
                        sigma_j = sigma_in_j * 1/2 * (1 + z_j) * (1 - z_j)
                        sigmas_j.append(sigma_j)

                        deltas_v_nj = []
                        for neurona in self.red.capas[i-1].neuronas:
                            delta_v_nj = self.alpha * sigma_j * neurona.valor_salida
                            deltas_v_nj.append(delta_v_nj)
                        deltas_v.append(deltas_v_nj)
                    cambios_pesos.append(deltas_v)

                    sigmas_k = sigmas_j

                # Paso 9, actualizar pesos, para ello invertimos el orden de la lista cambios_pesos
                cambios_pesos.reverse()
                for n_capa in range(len(cambios_pesos)):
                    for n_neu_j in range(len(cambios_pesos[n_capa])):
                        for n_neu_i in range(len(cambios_pesos[n_capa][n_neu_j])):
                            self.red.capas[n_capa].neuronas[n_neu_i].conexiones[n_neu_j].peso += cambios_pesos[n_capa][n_neu_j][n_neu_i]

                # Paso 10, comprobar condicion de parada e inicial las entradas de ultima capa a 0
                self.red.capas[-1].inicializar()

            # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
            error_cuad_med /= len(y_train)
            if error_cuad_med > self.tolerancia:
                parar = False

            list_ecm.append(error_cuad_med)
            list_epoca.append(epoca)
            # print(f"Epoca: {epoca}, MSE: {error_cuad_med}")

            # Si existe datos de validacion, se evalua para obtener el score,
            # el mejor score empieza con 0 y se va cambiando
            # si el acierto en validacion es 5% inferior que mejor score, se para
            # si el acierto se repite 20 veces con mejor score, se para
            if len(X_val) != 0 and len(y_val) != 0:
                acierto = self.score(X_val, y_val)
                if acierto > mejor_score:
                    mejor_score = acierto
                    mejor_score_repetido = 0
                elif acierto < mejor_score:
                    if mejor_score - acierto > 3:
                        break
                elif acierto == mejor_score:
                    mejor_score_repetido += 1
                    if mejor_score_repetido == 20:
                        break

        return list_epoca, list_ecm

    # Funcion para la prediccion de la red del adaliene
    def test(self, X_test):
        # Vacia todas las entradas de la red
        self.red.inicializar()

        y_preds = []
        for x in X_test:
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.red.capas[0].neuronas) - 1):
                self.red.capas[0].neuronas[i+1].inicializar(x[i])
            
            for capa in self.red.capas[:-1]:
                capa.disparar()
                capa.inicializar()
                capa.propagar()

            # Obtención de las salidas Y, de la capa de salida
            self.red.capas[-1].disparar()
            self.red.capas[-1].inicializar()

            y_pred = []
            for neurona_k in self.red.capas[-1].neuronas:
                # y_pred.append(1 if neurona_k.valor_salida >= 0 else -1)
                y_pred.append(neurona_k.valor_salida)
            fix = np.zeros(len(self.red.capas[-1].neuronas))
            fix[np.argmax(y_pred)] = 1
            y_pred = [-1 if int(value) == 0 else int(value) for value in fix]
            y_preds.append(y_pred)
        
        return y_preds

    def fix_y(self, y):
        # Si la longitud de y es 1, ya es una salida binaria, 0 o 1
        # Aqui, Clase 1 = -1 y clase 2 = 1
        if len(y[0]) == 1:
            return y

        y_fix = []
        for y_class in y:
            y_fix.append(np.argmax(y_class))
        return y_fix
    
    def matriz_confusion(self, y_true, y_preds):
        fix_y_true = self.fix_y(y_true)
        fix_y_preds = self.fix_y(y_preds)
        return confusion_matrix(fix_y_true, fix_y_preds)

    # Funcion para la prediccion de la red del adaliene
    def test_write(self, X_test, f_out):
        y_preds = self.test(X_test)

        text = ''
        for i in range(len(self.red.capas[0].neuronas) - 1):
            text += "X{}\t".format(i+1)
        for j in range(len(self.red.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
        text += "\n"
        f_out.write(text)

        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(X_test)):
            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for x_i in X_test[index]:
                text += "{}\t".format(x_i)

            for y_i in y_preds[index]:
                text += "{:.2f}\t".format(y_i)

            text += '\n'
            f_out.write(text)

    # Funcion para imprimir el score de la red
    def score(self, X, y):
        y_preds = self.test(X)
        n_acierto = 0
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(y)):
            if (y[index] == y_preds[index]).all():
                n_acierto += 1
        
        # print("Porcentaje de aciertos: {}%\n".format(n_acierto/len(y)*100))
        return n_acierto/len(y)*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='red')
    parser.add_argument('--modo1',
                        nargs=2,
                        metavar=('fichero', 'porcion'),
                        help='Nombre del fichero de entrada y porcion')
    parser.add_argument('--modo2',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada')
    parser.add_argument('--modo3',
                        nargs=2,
                        metavar=('train_file', 'test_file'),
                        help='Nombre del fichero de entrada para train y para test')
    parser.add_argument('--f_out',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de salida')
    parser.add_argument('--alpha',
                        nargs=1,
                        metavar='alpha',
                        help='Tasa de aprendizaje de la red')
    parser.add_argument('--torelancia',
                        nargs=1,
                        metavar='torelancia',
                        help='Torelancia de la red')
    parser.add_argument('--neu',
                        nargs="+",
                        metavar='neu',
                        type=int,
                        help='Lista de neuronas por capa')
    parser.add_argument('--norm',
                        nargs=1,
                        metavar='norm',
                        choices=['true', 'false'],
                        help='True or false para normalizar los datos')
    parser.add_argument('--epoca',
                        nargs=1,
                        metavar='epoca',
                        help='Num de epoca de la red')

    args = parser.parse_args()

    if not args.f_out:
        f_out = stdout
    else:
        f_out = open(args.f_out[0], 'w')
    
    alpha = float(args.alpha[0]) if args.alpha else 0.1
    torelancia = float(args.torelancia[0]) if args.torelancia else 0.01
    epoca = int(args.epoca[0]) if args.epoca else 100
    capas_neu = args.neu if args.neu else []
    if args.norm:
        norm = args.norm[0] == 'true'
    else:
        norm = False

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1], norm=norm)
        red = Multicapa(alpha=alpha, capas_neu=capas_neu, tolerancia=torelancia, epoca=epoca)
        lista_epoca, lista_ecm = red.train(X_train, y_train, X_test, y_test)
        y_preds = red.test(X_test)
        print(f"Se ha ejecutado {len(lista_epoca)} epocas.")
        print("Porcentaje de aciertos: {}%\n".format(red.score(X_test, y_test)))
        print(red.matriz_confusion(y_test, y_preds))
        # red.test_write(X_test, y_test, f_out)

    elif args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0], norm=norm)
        red = Multicapa(alpha=alpha, capas_neu=capas_neu, tolerancia=torelancia, epoca=epoca)
        lista_epoca, lista_ecm = red.train(X, y)
        y_preds = red.test(X)
        print(f"Se ha ejecutado {len(lista_epoca)} epocas.")
        print("Porcentaje de aciertos: {}%\n".format(red.score(X, y)))
        print(red.matriz_confusion(y, y_preds))
        # red.test_write(X, y, f_out)
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1], norm=norm)
        lista_epoca, lista_ecm = red = Multicapa(alpha=alpha, capas_neu=capas_neu, tolerancia=torelancia, epoca=epoca)
        red.train(X_train, y_train)
        y_preds = red.test(X_test)
        print(f"Se ha ejecutado {len(lista_epoca)} epocas.")
        print("Porcentaje de aciertos: {}%\n".format(red.score(X_test, y_test)))
        print(red.matriz_confusion(y_test, y_preds))
        # red.test_write(X_test, y_test, f_out)
    else:
        print("Error en los argumentos, necesita especificar algun modo de operacion.")
        exit(1)
    
    f_out.close() if args.f_out else None

# plt.plot(list_epoca, list_ecm)
# plt.show()