import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

class LeerFichero:
    @staticmethod
    def mode1(file_name, porcion, norm=False):
        f = open(file_name, 'r')

        n_atr, n_class = f.readline().replace('\n', '').split()
        n_atr = int(n_atr)
        n_class = int(n_class)
        X, y = [], []
        for row in f.readlines():
            row = row.replace('\n', '').split()
            X.append(row[:-n_class])
            y_fix = [-1 if int(value) == 0 else int(value) for value in row[-n_class:]]
            y.append(y_fix)
        f.close()

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(porcion))

        if norm:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def mode2(file_name, norm=False):
        f = open(file_name, 'r')

        n_atr, n_class = f.readline().replace('\n', '').split()
        n_atr = int(n_atr)
        n_class = int(n_class)
        X, y = [], []
        for row in f.readlines():
            row = row.replace('\n', '').split()
            X.append(row[:-n_class])
            y_fix = [-1 if int(value) == 0 else int(value) for value in row[-n_class:]]
            y.append(y_fix)
        f.close()

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        if norm:
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

        return X, y

    @staticmethod
    def mode3(f_train_name, f_test_name, norm=False):
        f_train = open(f_train_name, 'r')
        f_test = open(f_test_name, 'r')

        n_atr, n_class = f_train.readline().replace('\n', '').split()
        n_atr = int(n_atr)
        n_class = int(n_class)
        X_train, y_train = [], []
        for row in f_train.readlines():
            row = row.replace('\n', '').split()
            X_train.append(row[:-n_class])
            y_fix = [-1 if int(value) == 0 else int(value) for value in row[-n_class:]]
            y.append(y_fix)
        
        n_atr, n_class = f_test.readline().replace('\n', '').split()
        n_atr = int(n_atr)
        n_class = int(n_class)
        X_test, y_test = [], []
        for row in f_test.readlines():
            row = row.replace('\n', '').split()
            X_test.append(row[:-n_class])
            y_fix = [-1 if int(value) == 0 else int(value) for value in row[-n_class:]]
            y_test.append(y_fix)
            y_test.append(row[-n_class:])

        f_train.close()
        f_test.close()

        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=int)
        X_test = np.array(X_test, dtype=float)
        y_test = np.array(y_test, dtype=int)

        if norm:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

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
