from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from leerFichero import LeerFichero
import matplotlib.pyplot as plt

class Perceptron():

    # Funcion para la construccion de la red del perceptron
    def __init__(self, umbral=0.0, alpha=1.0, epoca=100):
        self.perceptron = RedNeuronal()
        self.umbral = umbral
        self.alpha = alpha
        self.epoca = epoca

    def make_red(self, n_atributos, n_clases):
         # Neuronas capa entrada
        neuronas_entrada = []
        for i in range(0, n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.DIRECTA))
        # Neurona correspondiente al bias
        neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.SESGOIGUAL))

        # Neuronas capa salida
        neuronas_salida = [] 
        for i in range(n_clases):
            neuronas_salida.append(Neurona(umbral = self.umbral, tipo=Tipo.PERCEPTRON))  # Se aniade el umbral especificado
        
        # Creacion capas y anidamiento de neuronas en estas, y anidamiento capas dentro de la red neuronal perceptron
        capa_entrada = Capa()
        capa_salida = Capa()
        for i in range(n_atributos + 1): # + 1 debido a que en la capa de entrada se aniade tambien el bias
            capa_entrada.aniadir(neuronas_entrada[i])
        for i in range(n_clases): 
            capa_salida.aniadir(neuronas_salida[i])
        self.perceptron.aniadir(capa_entrada)
        self.perceptron.aniadir(capa_salida)

        # Conexiones entre capas. -1 porque el ultimo no tiene conexiones 
        for i in range(len(self.perceptron.capas) - 1):
            self.perceptron.capas[i].conectar(self.perceptron.capas[i+1], 0 , 0)

    # Funcion para la realizacion del entrenamiento por el perceptron
    def train(self, X_train, y_train):
        # Se crea la red del perceptron
        self.make_red(X_train.shape[1], y_train.shape[1])

        # Uso plot
        X = []
        Y = []

        # Paso 0, inicial todos los pesos y sesgo
        self.perceptron.inicializar()
        epoca = 0

        # Paso 1, mientras que haya actualizacion de peso, se ejecutra paso 2-6
        parar = False
        while not parar and epoca < self.epoca:
            # Se pone parar a True suponiendo que no va a actualizar durante la epoca
            parar = True

            # Uso para el calculo del error cuadratico medio en cada epoca
            epoca += 1
            error_cuad_med = 0

            # Saca los pesos de la conexion y los guarda en una lista,
            # uso para comparar si los pesos han sido actualizado
            # La lista seria [[w11, w12, w13], [w21, w22, w23], [w31, w32, w33]]
            ultimo_pesos = []
            for j in range(len(self.perceptron.capas[-1].neuronas)):
                temp = []
                for neurona in self.perceptron.capas[0].neuronas:
                    temp.append(neurona.conexiones[j].peso)
                ultimo_pesos.append(temp)

            # Paso 2, para cada par de entrenamiento, ejecutar paso 3-5
            for m in range(len(y_train)):
                record_x = X_train[m]
                record_y = y_train[m]

                # Paso 3, establecer las activaciones a las neuronas de entrada, a excepcion del bias 
                for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                    self.perceptron.capas[0].neuronas[i].inicializar(record_x[i])
                
                # Paso 4, calcular la respuesta de cada neurona de salida y_in
                self.perceptron.capas[0].disparar()
                self.perceptron.capas[0].inicializar()
                self.perceptron.capas[0].propagar()

                # Obtención de las salidas f(y_in), de la capa de salida
                self.perceptron.capas[-1].disparar()
                self.perceptron.capas[-1].inicializar()
                
                # Calcular error cuadratico medio wij = Sumatorio(t_j - y_in_j)^2
                error = False   # flag para saber si hay diferencia del valor predecido respecto al real
                for neurona, t in zip(self.perceptron.capas[-1].neuronas, record_y):
                    error_cuad_med += (t - neurona.valor_salida)**2
                    if neurona.valor_salida != t:
                        error = True
                
                # Paso 5, ajustar los pesos en el caso de que alguna clase no coincide
                if error:
                    # Paso 5.a Ajuste de los pesos menos bias
                    for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                        neurona_i = self.perceptron.capas[0].neuronas[i]
                        # Conexiones de la neurona en cuestion
                        for j in range(len(self.perceptron.capas[-1].neuronas)):
                            nuevo_peso = neurona_i.conexiones[j].peso_anterior + self.alpha * record_y[j] * record_x[i]
                            neurona_i.conexiones[j].peso = nuevo_peso
                            neurona_i.conexiones[j].peso_anterior = nuevo_peso

                            # Si hay algun cambio en los pesos respecto anterior, se actualiza el flag
                            if ultimo_pesos[j][i] != nuevo_peso:
                                parar = False

                    # Paso 5.b Ajuste de los pesos en bias
                    bias_i = self.perceptron.capas[0].neuronas[i+1]
                    for j in range(len(self.perceptron.capas[-1].neuronas)):
                        nuevo_peso = bias_i.conexiones[j].peso_anterior + self.alpha * record_y[j]
                        bias_i.conexiones[j].peso = nuevo_peso
                        bias_i.conexiones[j].peso_anterior = nuevo_peso

                        # Si hay algun cambio en los pesos respecto anterior, se actualiza el flag
                        if ultimo_pesos[j][i] != nuevo_peso:
                            parar = False
            
            # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
            error_cuad_med = error_cuad_med/(len(y_train))
            
            # Impresion por pantalla de epoca completada y error cuadratico medio
            # print(f"Epoca: {epoca}, MSE: {error_cuad_med}")
            X.append(epoca)
            Y.append(error_cuad_med)
            
            # Paso 6, si peso_actualizado = False, se termina el entrenamiento, sino vuelve al bucle while
        """
        plt.plot(X, Y)
        plt.show()
        """
    # Funcion para la prediccion de la red del perceptron
    def test(self, X_test, y_test, f_out):
        text = ""
        for i in range(len(self.perceptron.capas[0].neuronas) - 1):
            text += "X{}\t".format(i+1)
        for j in range(len(self.perceptron.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
        text += "\n"
        f_out.write(text)

        # Vacia todas las entradas de la red
        self.perceptron.inicializar()

        n_acierto = 0
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(X_test)):
            x = X_test[index]
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                self.perceptron.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.perceptron.capas[0].disparar()
            self.perceptron.capas[0].inicializar()
            self.perceptron.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.perceptron.capas[-1].disparar()
            self.perceptron.capas[-1].inicializar()

            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for x_i in x:
                text += "{}\t".format(x_i)

            error = False
            for j in range(len(self.perceptron.capas[-1].neuronas)):
                y_in = self.perceptron.capas[-1].neuronas[j].valor_salida
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
    
    # Funcion para imprimir el score de la red del perceptron
    def score(self, X, y):
        # Vacia todas las entradas de la red
        self.perceptron.inicializar()

        n_acierto = 0
        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for index in range(len(X)):
            x = X[index]
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                self.perceptron.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.perceptron.capas[0].disparar()
            self.perceptron.capas[0].inicializar()
            self.perceptron.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.perceptron.capas[-1].disparar()
            self.perceptron.capas[-1].inicializar()


            error = False
            for j in range(len(self.perceptron.capas[-1].neuronas)):
                y_in = self.perceptron.capas[-1].neuronas[j].valor_salida

                if y_in != y[index][j]:
                    error = True
            
            if not error:
                n_acierto += 1
        
        print("Porcentaje de aciertos: {}%\n".format(n_acierto/len(y)*100))

    def get_weights(self):
        text = ""
        for j in range(len(self.perceptron.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
            for i in range(len(self.perceptron.capas[0].neuronas)):
                text += "W{}: {:.4f}\t".format(i+1, self.perceptron.capas[0].neuronas[i].conexiones[j].peso)
            text += "\n"
        return text

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
                        metavar=('train_file', 'test_file'),
                        help='Nombre del fichero de entrada para train y para test')
    parser.add_argument('--f_out',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de salida')
    parser.add_argument('--umbral',
                        nargs=1,
                        metavar='umbral',
                        help='Umbral de la red')
    parser.add_argument('--alpha',
                        nargs=1,
                        metavar='alpha',
                        help='Tasa de aprendizaje de la red')
    parser.add_argument('--epoca',
                        nargs=1,
                        metavar='epoca',
                        help='Num de epoca de la red')

    args = parser.parse_args()

    if not args.f_out:
        f_out = stdout
    else:
        f_out = open(args.f_out[0], 'w')
    
    umbral = float(args.umbral[0]) if args.umbral else 0.2
    alpha = float(args.alpha[0]) if args.alpha else 1
    epoca = int(args.epoca[0]) if args.epoca else 100

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        perceptron = Perceptron(umbral=umbral, alpha=alpha, epoca=epoca)
        perceptron.train(X_train, y_train)
        # perceptron.score(X_train, y_train)
        perceptron.test(X_test, y_test, f_out)

    elif args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0])
        perceptron = Perceptron(umbral=umbral, alpha=alpha, epoca=epoca)
        perceptron.train(X, y)
        # perceptron.score(X, y)
        perceptron.test(X, y, f_out)
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        perceptron = Perceptron(umbral=umbral, alpha=alpha, epoca=epoca)
        perceptron.train(X_train, y_train)
        # perceptron.score(X_train, y_train)
        perceptron.test(X_test, y_test, f_out)
    else:
        print("Error en los argumentos, necesita especificar algun modo de operacion.")
        exit(1)
    
    f_out.close() if args.f_out else None
