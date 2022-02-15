from sympy import arg
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from leerFichero import LeerFichero

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
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        exit(0)

    if args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0])
        exit(0)

    if args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        exit(0)
