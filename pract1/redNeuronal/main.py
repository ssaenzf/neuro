from RedNeuronal import RedNeuronal
from Capa import Capa
from Neurona import Neurona
from Conexion import Conexion
from Tipo import Tipo

fichero_entrada = input("Introduzca el nombre del fichero de entrada: ")
fichero_salida = input("Introduzca el nombre del fichero de salida: ")

red_McCulloch = RedNeuronal()

capa_entrada = Capa()
x1 = Neurona(umbral = 0, tipo=Tipo.DIRECTA)
x2 = Neurona(umbral = 0, tipo=Tipo.DIRECTA)

capa_oculta = Capa()
z1 = Neurona(umbral = 2, tipo=Tipo.MCCULLOCH)
z2 = Neurona(umbral = 2, tipo=Tipo.MCCULLOCH)

capa_salida = Capa()
y1 = Neurona(umbral = 2, tipo=Tipo.MCCULLOCH)
y2 = Neurona(umbral = 2, tipo=Tipo.MCCULLOCH)


x1.Conectar(y1, 2)
x2.Conectar(z1, -1)
x2.Conectar(z2, 2)
x2.Conectar(y2, 1)

z1.Conectar(y1, 2)
z2.Conectar(z1, 2)
z2.Conectar(y2, 1)

capa_entrada.aniadir(x1)
capa_entrada.aniadir(x2)

capa_oculta.aniadir(z1)
capa_oculta.aniadir(z2)

capa_salida.aniadir(y1)
capa_salida.aniadir(y2)

red_McCulloch.aniadir(capa_entrada)
red_McCulloch.aniadir(capa_oculta)
red_McCulloch.aniadir(capa_salida)

red_McCulloch.inicializar()

handler_fichero_salida = open(fichero_salida, 'w')
handler_fichero_entrada = open(fichero_entrada, 'r')
handler_fichero_salida.write('x1  x2  z1  z2  y1  y2')

for row in handler_fichero_entrada.readlines():
    x1.Inicializar(row[0])
    x2.Inicializar(row[2])