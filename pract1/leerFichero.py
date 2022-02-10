from sympy import arg
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse

def mode1(file_name, porcion):
    f = open(file_name, 'r')

    n_atr, n_class = f.readline().replace('\n', '').split(' ')
    print(n_atr, n_class)
    for row in f.readlines():
        row = row.replace('\n', '').split(' ')
        print(row)

    f.close()

def mode2(file_name):
    f = open(file_name, 'r')

    n_atr, n_class = f.readline().replace('\n', '').split(' ')
    print(n_atr, n_class)
    for row in f.readlines():
        row = row.replace('\n', '').split(' ')
        print(row)

    f.close()

def mode3(f_train_name, f_test_name):
    f_train = open(f_train_name, 'r')
    f_test = open(f_test_name, 'r')

    n_atr, n_class = f_train.readline().replace('\n', '').split(' ')
    print(n_atr, n_class)
    for row in f_train.readlines():
        row = row.replace('\n', '').split(' ')
        print(row)

    f_train.close()
    f_test.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Red neuronal McCulloch-Pitts')
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
        mode1(args.modo1[0], args.modo1[1])
        exit(0)
    
    if args.modo2:
        mode2(args.modo2[0])
        exit(0)
    
    if args.modo3:
        mode3(args.modo3[0], args.modo3[1])
        exit(0)
