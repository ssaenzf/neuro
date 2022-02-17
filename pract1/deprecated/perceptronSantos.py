from sympy import arg
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from leerFichero import LeerFichero

class Perceptron():

    # Funcion para la construccion de la red del perceptron
    def __init__(self, X_train, y_train, epocas=100, umbral=0.0, alpha=1.0):
        self.epocas = epocas
        self.perceptron = RedNeuronal()
        self.alpha = alpha
        self.X_train = X_train
        self.y_train = y_train
        n_atributos = len(X_train[0])
        n_clases = len(y_train[0])

        # Neuronas capa entrada
        neuronas_entrada = []
        for i in range(0, n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.DIRECTA))
        # Neurona correspondiente al bias
        neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.SESGO))

        # Neuronas capa salida
        neuronas_salida = [] 
        for i in range(n_clases):
            neuronas_salida.append(Neurona(umbral = umbral, tipo=Tipo.PERCEPTRON))  # Se aniade el umbral especificado
        
        # Conexiones. Todas las neuronas de la capa de entrada se conectan con todas las neuronas de la capa de salida
        for i in range(n_atributos + 1): # + 1 debido a que hay que conectar el bias a todas las neuronas de la capa de salida tambien
            for j in range(n_clases):
                neuronas_entrada[i].conectar(neuronas_salida[j], 0) # Pesos de las conexiones inicialmente a 0

        # Creacion capas y anidamiento de neuronas en estas, y anidamiento capas dentro de la red neuronal perceptron
        capa_entrada = Capa()
        capa_salida = Capa()
        for i in range(n_atributos + 1): # + 1 debido a que en la capa de entrada se aniade tambien el bias
            capa_entrada.aniadir(neuronas_entrada[i])
        for i in range(n_clases): 
            capa_salida.aniadir(neuronas_salida[i])
        self.perceptron.aniadir(capa_entrada)
        self.perceptron.aniadir(capa_salida)

    # Funcion para la realizacion del entrenamiento por el perceptron
    def train(self):
        for epoca in range(self.epocas + 1):
            error_cuad_med = 0
            for m in range(len(self.y_train)):
                record_x = self.X_train[m]
                record_y = self.y_train[m]
                # Se inicializan a 0 las neuronas de la capa de salida
                for i in range(len(self.perceptron.capas[1].neuronas)):
                    self.perceptron.capas[1].neuronas[i].inicializar(0.0)
                
                # Se inicializan con los valores de entrada las neuronas de la capa de entrada, a excepcion del bias 
                for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                    self.perceptron.capas[0].neuronas[i].inicializar(float(record_x[i]))
                    self.perceptron.capas[0].neuronas[i].disparar()
                    self.perceptron.capas[0].neuronas[i].propagar()
                self.perceptron.capas[0].neuronas[i+1].inicializar(0.0) # Neurona del bias, siempre inicializada a 1
                self.perceptron.capas[0].neuronas[i+1].disparar()
                self.perceptron.capas[0].neuronas[i+1].propagar()

                # Obtención de las salidas, de la capa de salida
                for i in range(len(self.perceptron.capas[1].neuronas)):
                    self.perceptron.capas[1].neuronas[i].disparar()
                
                # Obtencion flag sobre si ha habido error o no en la prediccion para saber realizar el posterior ajuste 
                # de pesos a las conexiones. Ademas obtencion del error cuadratico medio
                error = False   # flag para saber si hay diferencia del valor predecido respecto al real
                for i in range(len(self.perceptron.capas[1].neuronas)):
                    error_cuad_med += (self.perceptron.capas[1].neuronas[i].valor_salida - float(record_y[i]))**2
                    if self.perceptron.capas[1].neuronas[i].valor_salida != float(record_y[i]):
                        error = True
                
                # Ajuste de pesos en caso de error
                if error == True:
                    for i in range(len(self.perceptron.capas[0].neuronas) - 1):  # -1 debido a que el bias se ajusta de forma distinta y por tanto separada
                        for j in range(len(self.perceptron.capas[0].neuronas[i].conexiones)): # Conexiones de la neurona en cuestion
                            # Cada conexion esta conectada a una neurona de salida que se debera encontrar su indice, para saber por otra parte
                            # el indice que se corresponde del record_y del y_train
                            for k in range(len(self.perceptron.capas[1].neuronas)):
                                if self.perceptron.capas[0].neuronas[i].conexiones[j].neurona == self.perceptron.capas[1].neuronas[k]:
                                    # peso nuevo = peso anterior + alpha*t*xi
                                    self.perceptron.capas[0].neuronas[i].conexiones[j].peso_anterior = self.perceptron.capas[0].neuronas[i].conexiones[j].peso
                                    self.perceptron.capas[0].neuronas[i].conexiones[j].peso = self.perceptron.capas[0].neuronas[i].conexiones[j].peso + float(record_y[k]) * self.alpha * self.perceptron.capas[0].neuronas[i].valor_salida
                    # Ajuste peso conexion bias
                    # Se busca la neurona que esta conectada a la conexion del bias
                    for k in range(len(self.perceptron.capas[0].neuronas[i+1].conexiones)):
                        for j in range(len(self.perceptron.capas[1].neuronas)):
                            if self.perceptron.capas[0].neuronas[i+1].conexiones[k].neurona == self.perceptron.capas[1].neuronas[j]:
                                self.perceptron.capas[0].neuronas[i+1].conexiones[k].peso_anterior = self.perceptron.capas[0].neuronas[i+1].conexiones[k].peso
                                # peso nuevo = peso anterior + alpha*t
                                self.perceptron.capas[0].neuronas[i+1].conexiones[k].peso = self.perceptron.capas[0].neuronas[i+1].conexiones[k].peso + self.alpha * float(record_y[j])
            
            # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
            error_cuad_med = error_cuad_med/(len(record_y)*len(self.y_train))
            
            # Impresion por pantalla de epoca completada y error cuadratico medio
            print(f"Epoca: {epoca}/{self.epocas}, MSE: {error_cuad_med}")
    
    # Funcion para la prediccion de la red del perceptron
    def predecir(self, X_test, fichero_salida):
        # Se abre el fichero donde se escriben las predicciones resueltas
        f_out = open(fichero_salida, 'w')
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for x in X_test:
            # Las neuronas de salida se inicializan con valor 0, para que no se acumulen valores aqui anteriores
            for i in range(len(self.perceptron.capas[1].neuronas)):
                self.perceptron.capas[1].neuronas[i].inicializar(0.0)
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                self.perceptron.capas[0].neuronas[i].inicializar(float(x[i]))
                self.perceptron.capas[0].neuronas[i].disparar()
                self.perceptron.capas[0].neuronas[i].propagar()
            self.perceptron.capas[0].neuronas[i+1].disparar()
            self.perceptron.capas[0].neuronas[i+1].propagar()
            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for i in range(len(self.perceptron.capas[1].neuronas)):
                self.perceptron.capas[1].neuronas[i].disparar()
                text += f" {self.perceptron.capas[1].neuronas[i].valor_salida}"
            text += '\n'
            f_out.write(text)
        f_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perceptron')
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
                        metavar=('train', 'test'),
                        help='Nombre del fichero de entrada')

    args = parser.parse_args()

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        perceptron = Perceptron(X_train, y_train, epocas=1000, umbral=0.2, alpha=0.1)
        perceptron.train()
        perceptron.predecir(X_test, "salida.txt")

    elif args.modo2:
        X_train, y_train = LeerFichero.mode2(args.modo2[0])
        perceptron = Perceptron(X_train, y_train, epocas=1000, umbral=0.1, alpha=0.01)
        perceptron.train()
        perceptron.predecir(X_train, "salida.txt")
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        perceptron = Perceptron(X_train, y_train, epocas=1000, umbral=0.2, alpha=0.1)
        perceptron.train()
        perceptron.predecir(X_test, "salida.txt")
