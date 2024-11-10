import random

import openml
from openml.datasets import list_datasets
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
import pandas as pd

import out_path
import main


def get_small_datasets(max_samples=5000):
    # 获取数据集列表
    datasets = openml.datasets.list_datasets(output_format="dataframe")

    # 筛选出样本数小于 max_samples 且为分类任务的数据集，同时排除目标列存在缺失值的数据集
    small_classification_datasets = datasets[(datasets['NumberOfInstances'] <= max_samples) &
                                             (datasets['NumberOfInstances'] > 0) &
                                             (datasets['NumberOfClasses'] > 1) &
                                             (datasets['NumberOfMissingValues'] == 0)]

    return small_classification_datasets[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']]


def q_c_random_down():
    small_datasets = get_small_datasets()
    while True:
        random_dataset = small_datasets.sample(1).iloc[0]
        random_id = random_dataset['did']
        print(f"Attempting Dataset ID: {random_id}")

        df = fetch_openml(data_id=random_id, as_frame=True)

        # 检查是否有 target 列
        if not hasattr(df, 'target'):
            print(f"Dataset ID {random_id} does not have a target column. Trying another dataset...")
            continue  # 继续尝试下一个数据集

        # 获取目标列名称
        target_column = df.target.name
        if target_column not in df.data.columns:
            print(f"Target column '{target_column}' not found in Dataset ID {random_id}. Trying another dataset...")
            continue  # 继续尝试下一个数据集

        print(f"Using Dataset ID: {random_id} with target column '{target_column}'")
        return random_id, target_column  # 找到合适的数据集后返回


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


    if flag == "q_c22222":
        random_id, target_column = q_c_random_down()

        df = fetch_openml(data_id=random_id, as_frame=True)

        # 检查目标列的缺失值并处理
        if df.data[target_column].isnull().any():
            print(f"Dataset ID {random_id} has missing values in the target column.")
            df.data = df.data.dropna(subset=[target_column])
            print("Dropped rows with missing target values.")

        # 分离特征和目标，确保转换为 DataFrame
        X = pd.DataFrame(df.data.drop(columns=[target_column]))
        y = pd.DataFrame(df.data[target_column])

        # 检查 X 和 y 的结构
        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")

        # 输出到文件
        try:
            X.to_csv(f"{out_path.OutPath.X_path(flag)}", index=False)
            y.to_csv(f"{out_path.OutPath.y_path(flag)}", index=False)
            print("Files have been saved successfully.")
        except Exception as e:
            print(f"Failed to save files: {e}")

        return 0

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

