import os
import pandas as pd
import hashlib
from ucimlrepo import fetch_ucirepo  # 确认导入 fetch_ucirepo 函数
import time

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  The folder exists!  ---")

file = "../data"
mkdir(file)
# 缓存文件的路径
local_file = '../data/abalone_data.csv'
hash_file = '../data/abalone_hash.txt'

# 计算数据集的哈希值，用于更新检测
def compute_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

# 保存数据及其哈希值到本地
def save_data(df):
    if df is None:
        print("Error: The dataset is None. Cannot save to CSV.")
        return
    df.to_csv(local_file, index=False)  # 保存数据到本地 CSV
    hash_value = compute_hash(df)
    with open(hash_file, 'w') as f:
        f.write(hash_value)  # 保存数据集的哈希值

# 检查是否有新版本数据集（基于哈希值）
def is_dataset_updated(df):
    new_hash = compute_hash(df)
    if not os.path.exists(hash_file):
        return True  # 没有哈希文件，说明是第一次下载
    with open(hash_file, 'r') as f:
        old_hash = f.read()
    return new_hash != old_hash  # 如果哈希值不同，说明数据集有更新

# 从缓存或重新下载数据集
def load_or_update_dataset():
    # 检查是否有本地文件
    if os.path.exists(local_file):
        print("Loading data from local file...")
        df_local = pd.read_csv(local_file)  # 从本地文件加载数据

        # 尝试在线获取数据，如果网络连接失败，则使用本地数据
        try:
            print("Checking for updates...")
            df_online = fetch_ucirepo(id=1)['data']  # 从在线源获取数据
            if df_online.original is None:
                print("Error: Failed to fetch online dataset.")
                return df_local  # 如果在线数据获取失败，使用本地数据
            if is_dataset_updated(df_online.original):
                print("Dataset has been updated, downloading new version...")
                save_data(df_online.original)
                return df_online.original
            else:
                print("No update, Local dataset is up to date.")
                return df_local
        except Exception as e:
            print(f"Error fetching online dataset: {e}")
            print("No internet connection, using local dataset...")
            return df_local
    else:
        # 如果没有本地文件，直接下载并缓存
        try:
            print("No local file found, downloading data...")
            df_online = fetch_ucirepo(id=1)['data']
            if df_online.original is None:
                print("Error: Failed to fetch online dataset.")
                return None  # 如果下载失败，返回 None
            save_data(df_online.data.original)
            return df_online.data.original
        except Exception as e:
            print(f"Error fetching online dataset: {e}")
            print("No internet connection and no local dataset available.")
            return None

def achieve_data_main():
    # 执行加载或更新逻辑
    abalone = load_or_update_dataset()
    if abalone is not None:
        print("\n" + "abalone head: ")
        print(abalone.head())
        print("\n")
    else:
        print("Failed to load the dataset.")

# if __name__ == "__main__":
#     achieve_data_main()
