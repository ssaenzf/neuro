from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from leerFichero import LeerFichero

class Adaline():

    # Funcion para la construccion de la red del perceptron
    def __init__(self, umbral=0.0, alpha=1.0, tolerancia=0.0):
        self.perceptron = RedNeuronal()
        self.umbral = umbral
        self.alpha = alpha
        self.tolerancia = tolerancia

    def make_red(self, n_atributos, n_clases):
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
            neuronas_salida.append(Neurona(umbral = self.umbral, tipo=Tipo.PERCEPTRON))  # Se aniade el umbral especificado
        
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
    def train(self, X_train, y_train):
        # Se crea la red del adaline
        self.make_red(X_train.shape[1], y_train.shape[1])

        # Paso 0, inicial todos los pesos y sesgo
        self.perceptron.inicializar()
        epoca = 0

        # Paso 1, mientras que haya actualizacion de peso, se ejecutra paso 2-6
        parar = False
        while not parar:
            # Se pone parar a True suponiendo que no va a actualizar durante la epoca
            parar = True

            # Uso para el calculo del error cuadratico medio en cada epoca
            epoca += 1
            error_cuad_med = 0

            # Inicializar cambio de peso a 0 para condicion de parada
            cambio_peso = 0

            # Paso 2, para cada par de entrenamiento, ejecutar paso 3-5
            for m in range(len(y_train)):
                record_x = X_train[m]
                record_y = y_train[m]

                # Paso 3, establecer las activaciones a las neuronas de entrada, a excepcion del bias 
                for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                    self.perceptron.capas[0].neuronas[i].inicializar(record_x[i])
                
                # Paso 4, calcular la respuesta de cada neurona de salida y_in
                self.perceptron.capas[0].disparar()
                self.perceptron.capas[0].propagar()

                # Obtención de las salidas y_in, de la capa de salida
                self.perceptron.capas[-1].disparar()
                
                # Obtencion del error cuadratico medio
                for neurona, t in zip(self.perceptron.capas[-1].neuronas, record_y):
                    error_cuad_med += (neurona.valor_salida - t)**2
                
                # Paso 5.a Ajuste de los pesos menos bias
                for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                    neurona_i = self.perceptron.capas[0].neuronas[i]
                    # Conexiones de la neurona en cuestion
                    for j in range(len(self.perceptron.capas[-1].neuronas)):
                        y_in = self.perceptron.capas[-1].neuronas[j].valor_salida
                        cambio = self.alpha * (record_y[j] - y_in) * record_x[i]
                        nuevo_peso = neurona_i.conexiones[j].peso_anterior + cambio
                        neurona_i.conexiones[j].peso = nuevo_peso
                        neurona_i.conexiones[j].peso_anterior = nuevo_peso

                        # Actualiza cambio peso si existe uno mayor
                        if cambio_peso < cambio:
                            cambio_peso = cambio

                # Paso 5.b Ajuste de los pesos en bias
                bias_i = self.perceptron.capas[0].neuronas[i+1]
                for j in range(len(self.perceptron.capas[-1].neuronas)):
                    y_in = self.perceptron.capas[-1].neuronas[j].valor_salida
                    cambio = self.alpha * (record_y[j] - y_in)
                    nuevo_peso = bias_i.conexiones[j].peso_anterior + cambio
                    bias_i.conexiones[j].peso = nuevo_peso
                    bias_i.conexiones[j].peso_anterior = nuevo_peso

                    # Actualiza cambio peso si existe uno mayor
                    if cambio_peso < cambio:
                            cambio_peso = cambio

            # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
            error_cuad_med = error_cuad_med/(len(record_y)*len(y_train))
            
            # Impresion por pantalla de epoca completada y error cuadratico medio
            print(f"Epoca: {epoca}, MSE: {error_cuad_med}")
            
            # Paso 6, si cambio_peso es menor que la torelancia, se termina, sino vuelve al bucle while
            if cambio_peso >= self.tolerancia:
                parar = False
    
    # Funcion para la prediccion de la red del perceptron
    def predecir(self, X_test, f_out):
        text = ""
        for i in range(len(self.perceptron.capas[0].neuronas) - 1):
            text += "X{}\t".format(i+1)
        for j in range(len(self.perceptron.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
        text += "\n"
        f_out.write(text)

        # Vacia todas las entradas de la red
        self.perceptron.inicializar()

        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for x in X_test:
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                self.perceptron.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.perceptron.capas[0].disparar()
            self.perceptron.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.perceptron.capas[-1].disparar()

            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for neurona in self.perceptron.capas[-1].neuronas:
                for x_i in x:
                    text += "{}\t".format(x_i)
                text += "{:.2f}\t".format(neurona.valor_salida)
            text += '\n'
            f_out.write(text)

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
    parser.add_argument('--f_out',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada')

    args = parser.parse_args()

    if not args.f_out:
        f_out = stdout
    else:
        f_out = open(args.f_out[0], 'w')

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        adaline = Adaline(umbral=0.2, alpha=0.1, tolerancia=0.1)
        adaline.train(X_train, y_train)
        adaline.predecir(X_test, f_out)

    elif args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0])
        adaline = Adaline(umbral=0.2, alpha=1, tolerancia=0.1)
        adaline.train(X, y)
        adaline.predecir(X, f_out)
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        adaline = Adaline(umbral=0.2, alpha=0.1, tolerancia=0.1)
        adaline.train(X_train, y_train)
        adaline.predecir(X_test, f_out)
    else:
        print("Error en los argumentos, necesita especificar algun modo de operacion.")
        exit(1)
    
    f_out.close() if args.f_out else None
