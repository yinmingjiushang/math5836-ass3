import pandas as pd

import out_path

def data_input(flag):
    X_path = out_path.OutPath.X_path(flag)
    y_path = out_path.OutPath.y_path(flag)

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    return X, y
