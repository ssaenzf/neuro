from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Red neuronal McCulloch-Pitts')
    parser.add_argument('--f_in',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada',
                        required=True)
    parser.add_argument('--f_out',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada')

    args = parser.parse_args()

    f_in = open(args.f_in[0], 'r')
    f_out = None

    if not args.f_out:
        stdout.write('x1 x2 z1 z2 y1 y2\n')
    else:
        f_out = open(args.f_out[0], 'w')
        f_out.write('x1 x2 z1 z2 y1 y2\n')

    red_McCulloch = RedNeuronal()

    capa_entrada = Capa()
    x1 = Neurona(umbral=2.0, tipo=Tipo.DIRECTA, name='x1')
    x2 = Neurona(umbral=2.0, tipo=Tipo.DIRECTA, name='x2')

    capa_oculta = Capa()
    z1 = Neurona(umbral=2.0, tipo=Tipo.MCCULLOCH, name='z1')
    z2 = Neurona(umbral=2.0, tipo=Tipo.MCCULLOCH, name='z2')

    capa_salida = Capa()
    y1 = Neurona(umbral=2.0, tipo=Tipo.MCCULLOCH, name='y1')
    y2 = Neurona(umbral=2.0, tipo=Tipo.MCCULLOCH, name='y2')

    x1.conectar(y1, 2)
    x2.conectar(z1, -1)
    x2.conectar(z2, 2)
    x2.conectar(y2, 1)

    z1.conectar(y1, 2)
    z2.conectar(z1, 2)
    z2.conectar(y2, 1)

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
    for row in f_in.readlines():
        row = row.replace('\n', '').split(' ')
        x1.inicializar(int(row[0]))
        x2.inicializar(int(row[1]))

        red_McCulloch.disparar()
        red_McCulloch.inicializar()
        red_McCulloch.propagar()
        
        text = '{}  {}  {}  {}  {}  {}\n'.format(x1.valor_salida, x2.valor_salida, z1.valor_salida, z2.valor_salida, y1.valor_salida, y2.valor_salida)
        # print(text)
        f_out.write(text) if f_out else stdout.write(text)

    f_in.close()
    f_out.close() if f_out else None