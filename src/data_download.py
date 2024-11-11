import random

from ucimlrepo import fetch_ucirepo
import pandas as pd

import out_path
import main



def data_download(flag):
    if flag == "q_a":
        df = fetch_ucirepo(id=1)

    if flag == "q_b":
        df = fetch_ucirepo(id=30)

    if flag == "q_c":
        # download_random_uci_dataset()
        df = fetch_ucirepo(id=45)



    # 其他情况
    X = pd.DataFrame(df.data.features)
    y = pd.DataFrame(df.data.targets)

    # 输出到文件
    try:
        X.to_csv(f"{out_path.OutPath.X_path(flag)}", index=False)
        y.to_csv(f"{out_path.OutPath.y_path(flag)}", index=False)
        print("Files have been saved successfully.")
    except Exception as e:
        print(f"Failed to save files: {e}")

