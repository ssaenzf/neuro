from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from leerFichero import LeerFichero
import matplotlib.pyplot as plt

class Multicapa():

    # Funcion para la construccion de la red del red
    def __init__(self, alpha=1.0, tolerancia=0.0, epoca=100):
        self.red = RedNeuronal()
        self.alpha = alpha
        self.tolerancia = tolerancia
        self.epoca = epoca

    # TODO: cambiar esto que acepta una lista de nº de neuronas para cada capa oculta
    def make_red(self, n_atributos, n_clases):
        # Capa entrada
        capa_entrada = Capa()
        # Neurona correspondiente al bias
        capa_entrada.aniadir(Neurona(tipo=Tipo.SESGO))
        for i in range(n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            capa_entrada.aniadir(Neurona(tipo=Tipo.DIRECTA))
        
        # Capas ocultas
        capas_ocultas = []
        for i in range(2):
            capa_oculta = Capa()
            capa_oculta.aniadir(Neurona(tipo=Tipo.SESGO))
            for i in range(n_atributos):
                capa_oculta.aniadir(Neurona(tipo=Tipo.SIGMOIDE))
            capas_ocultas.append(capa_oculta)

        # Capa salida
        capa_salida = Capa()
        for i in range(n_clases):
            capa_salida.aniadir(Neurona(tipo=Tipo.SIGMOIDE))
        
        # Se añaden las capas a la red
        self.red.aniadir(capa_entrada)
        for capa_oculta in capas_ocultas:
            self.red.aniadir(capa_oculta)
        self.red.aniadir(capa_salida)

        # Conexiones entre capas. -1 porque el ultimo no tiene conexiones 
        for i in range(len(self.red.capas) - 1):
            self.red.capas[i].conectar(self.red.capas[i+1], -0.5 , 0.5)

    # Funcion para la realizacion del entrenamiento por el red
    def train(self, X_train, y_train):
        # Se crea la red del adaline
        self.make_red(X_train.shape[1], y_train.shape[1])

        # # Uso plot
        # X = []
        # Y = []

        # Paso 0, inicial todos los pesos y sesgo
        self.red.inicializar()
        epoca = 0

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
                    self.red.capas[0].neuronas[i].inicializar(record_x[i])
                
                # Paso 4, calcular la respuesta de las neuronas de las capas ocultas
                for capa in self.red.capas[:-1]:
                    capa.disparar()
                    capa.inicializar()
                    capa.propagar()

                # Paso 5, calcular la respuesta de las neuronas de la capa de salida
                # No se inicializa esta capa del momento, ya que necesita utilizar y_in para ajustar los pesos
                self.red.capas[-1].disparar()

                # Preparar lista inversa
                # Si list = [1,2,3,4,5], list[-1::-1] quiere decir desde ele -1, contar hacia delante, osea devuelve 5,4,3,2,1
                # capas_retro = self.red.capas[::-1]

                # Prepara lista para almacenar los cambios de peso
                cambios_pesos = []

                # Paso 6, cada neurona de salida Yk recibe un patron objetivo que corresponde al patron de entrada
                sigmas_k = []
                deltas_w = []
                capa = self.red.capas[-1]
                for k in range(len(capa.neuronas)):
                    y_k = capa.neuronas[k].valor_salida
                    sigma_k = (record_y[k] - y_k) * 1/2 *  (1 + y_k) * (1 - y_k)
                    sigmas_k.append(sigma_k)

                    # Primera Capa oculta contanto desde atras
                    for neurona in self.red.capas[-2].neuronas:
                        # delta_w_jk es el peso de la neurona de entrada j
                        # para neurona[0] que es bias, la salida es 1
                        delta_w_jk = self.alpha * sigma_k * neurona.valor_salida
                        deltas_w.append(delta_w_jk)
                cambios_pesos.append(deltas_w)

                # Paso 7 y 8, retropropagacion hacia atras de X capas hasta las conexiones de capa de entrada
                for i in range(len(self.red.capas) - 2, 0, -1):
                    sigmas_j = []
                    deltas_v = []
                    capa = self.red.capas[i]
                    for j in range(len(capa.neuronas)):
                        sigma_in_j = 0
                        for k in range(len(sigmas_k)):
                            sigma_in_j += sigmas_k[k] * capa.neuronas[j].pesos[k]
                        
                        z_j = capa.neuronas[j].valor_salida
                        sigma_j = sigma_in_j * 1/2 * (1 + z_j) * (1 - z_j)
                        sigmas_j.append(sigma_j)

                        for neurona in self.red.capas[i-1].neuronas:
                            delta_v_nj = self.alpha * sigma_j * neurona.valor_salida
                            deltas_v.append(delta_v_nj)
                    cambios_pesos.append(deltas_v)

                    sigmas_k = sigmas_j

                # Paso 9, actualizar pesos, para ello invertimos el orden de la lista cambios_pesos
                cambios_pesos.reverse()
                for i in range(len(cambios_pesos)):
                    for neurona in self.red.capas[i].neuronas:
                        for j in range(len(cambios_pesos[i])):
                            neurona.conexion[j].peso += cambios_pesos[i][j]
                
                # Paso 10, comprobar condicion de parada e inicial las entradas de ultima capa a 0
                self.red.capas[-1].inicializar()

        #     # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
        #     error_cuad_med = error_cuad_med/(len(y_train))
            
        #     # Impresion por pantalla de epoca completada y error cuadratico medio
        #     # print(f"Epoca: {epoca}, MSE: {error_cuad_med}")
        #     X.append(epoca)
        #     Y.append(error_cuad_med)
            
        #     # Paso 6, si cambio_peso es menor que la torelancia, se termina, sino vuelve al bucle while
        #     if cambio_peso >= self.tolerancia:
        #         parar = False
        # plt.plot(X, Y)
        # plt.show()

    # Funcion para la prediccion de la red del adaliene
    def test(self, X_test, y_test, f_out):
        text = ""
        for i in range(len(self.red.capas[0].neuronas) - 1):
            text += "X{}\t".format(i+1)
        for j in range(len(self.red.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
        text += "\n"
        f_out.write(text)

        # Vacia todas las entradas de la red
        self.red.inicializar()

        n_acierto = 0
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(X_test)):
            x = X_test[index]

            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.red.capas[0].neuronas) - 1):
                self.red.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.red.capas[0].disparar()
            self.red.capas[0].inicializar()
            self.red.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.red.capas[-1].disparar()
            self.red.capas[-1].inicializar()

            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for x_i in x:
                text += "{}\t".format(x_i)

            error = False
            for j in range(len(self.red.capas[-1].neuronas)):
                y_in = self.red.capas[-1].neuronas[j].valor_salida
                text += "{:.2f}\t".format(y_in)

                if y_in != y_test[index][j]:
                    error = True
            
            if not error:
                n_acierto += 1

            text += '\n'
            f_out.write(text)
        
        weights = self.get_weights()
        f_out.write(weights)
        f_out.write("Porcentaje de aciertos: {}%\n".format(n_acierto/len(y_test)*100))

    # Funcion para imprimir el score de la red del adaline
    def score(self, X, y):
        # Vacia todas las entradas de la red
        self.red.inicializar()

        n_acierto = 0
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(X)):
            x = X[index]
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.red.capas[0].neuronas) - 1):
                self.red.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.red.capas[0].disparar()
            self.red.capas[0].inicializar()
            self.red.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.red.capas[-1].disparar()
            self.red.capas[-1].inicializar()


            error = False
            for j in range(len(self.red.capas[-1].neuronas)):
                y_in = self.red.capas[-1].neuronas[j].valor_salida

                if y_in != y[index][j]:
                    error = True
            
            if not error:
                n_acierto += 1
        
        print("Porcentaje de aciertos: {}%\n".format(n_acierto/len(y)*100))
    
    def get_weights(self):
        text = ""
        for j in range(len(self.red.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
            for i in range(len(self.red.capas[0].neuronas)):
                text += "W{}: {:.5f}\t".format(i+1, self.red.capas[0].neuronas[i].conexiones[j].peso)
            text += "\n"
        return text

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
    parser.add_argument('--epoca',
                        nargs=1,
                        metavar='epoca',
                        help='Num de epoca de la red')

    args = parser.parse_args()

    if not args.f_out:
        f_out = stdout
    else:
        f_out = open(args.f_out[0], 'w')
    
    alpha = float(args.alpha[0]) if args.alpha else 0.3
    torelancia = float(args.torelancia[0]) if args.torelancia else 0.22
    epoca = int(args.epoca[0]) if args.epoca else 100

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        adaline = Multicapa(alpha=alpha, tolerancia=torelancia, epoca=epoca)
        adaline.train(X_train, y_train)
        # adaline.score(X_train, y_train)
        # adaline.test(X_test, y_test, f_out)

    elif args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0])
        adaline = Multicapa(alpha=alpha, tolerancia=torelancia, epoca=epoca)
        adaline.train(X, y)
        # adaline.score(X, y)
        # adaline.test(X, y, f_out)
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        adaline = Multicapa(alpha=alpha, tolerancia=torelancia, epoca=epoca)
        adaline.train(X_train, y_train)
        # adaline.score(X_train, y_train)
        # adaline.test(X_test, y_test, f_out)
    else:
        print("Error en los argumentos, necesita especificar algun modo de operacion.")
        exit(1)
    
    f_out.close() if args.f_out else None
