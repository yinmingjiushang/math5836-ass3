import random

from openml.datasets import list_datasets
from ucimlrepo import fetch_ucirepo
import pandas as pd

import out_path
import main



def download_random_uci_dataset():
    # 获取所有 UCI 数据集的列表并确保格式为 DataFrame
    datasets = list_datasets(output_format="dataframe")
    dataset_list = datasets.to_dict(orient="records")

    while True:
        # 从数据集中随机选择一个数据集
        random_dataset = random.choice(dataset_list)
        random_id = random_dataset["did"]
        print(f"Randomly selected Dataset ID: {random_id}, Name: {random_dataset['name']}")

        try:
            # 尝试下载数据集
            df = fetch_ucirepo(id=random_id)

            # 检查是否成功下载
            if df is None:
                print(f"Dataset ID {random_id} could not be loaded. Trying another dataset...")
                continue  # 重新选择一个数据集

            print(f"Successfully downloaded Dataset ID: {random_id}")

            # 检查是否有 target 列，分离特征和目标
            if 'target' in df.columns:
                target_column = 'target'
                X = df.drop(columns=[target_column])
                y = df[target_column]
                print(f"Dataset has a target column: '{target_column}'")
            else:
                X = df  # 没有 target 列则使用全部数据
                y = None
                print("Dataset does not have a target column")

            # 保存到 CSV 文件
            X.to_csv(f"X_{random_id}.csv", index=False)
            if y is not None:
                y.to_csv(f"y_{random_id}.csv", index=False)
            print(f"Files X_{random_id}.csv and y_{random_id}.csv have been saved.")
            break  # 成功下载并保存文件后退出循环

        except Exception as e:
            print(f"Failed to download Dataset ID {random_id}: {e}. Trying another dataset...")
            continue  # 重新选择另一个数据集


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

