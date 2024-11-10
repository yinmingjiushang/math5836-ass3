import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sympy import false


def data_clean(X, y):
    # 将所有非数值列转换为数值
    for column in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # 检查并转换 y 是否为数值类型（假设 y 是单列 DataFrame）
    if y.select_dtypes(include=['object', 'category']).shape[1] > 0:
        y = y.apply(lambda col: LabelEncoder().fit_transform(col))

    # 仅填充数值列的缺失值
    X[X.select_dtypes(include=['float64', 'int64']).columns] = X.select_dtypes(include=['float64', 'int64']).fillna(X.mean())

    return X, y


def map_values_to_bins(flag, y):
    if flag == "q_a":
        y = y.squeeze()
        age_bins = [-1, 7, 10, 15, float('inf')]
        age_labels = ['Class_1:0-7', 'Class_2:8-10', 'Class_3:11-15', 'Class_4:15-inf']
        yy = pd.cut(y, bins=age_bins, labels=False, right=True)

        return yy



def handle_imbalance(X, y, mode='weighted'):
    # 确保 y 是一维数组并转换为整数类型
    y = np.asarray(y).reshape(-1)  # 确保 y 是一维
    y = y.astype(int)  # 将 y 转换为整数类型

    # 检查 y 是否包含所有期望的标签
    unique_classes = np.unique(y).astype(int)  # 确保 unique_classes 是整数类型
    print(f"Unique classes in y: {unique_classes}")

    # 确保没有缺失值或异常标签
    if len(unique_classes) == 0:
        raise ValueError("y does not contain any valid labels.")

    class_weights = None

    if mode == 'oversample':
        # 使用 SMOTE 对少数类别进行过采样
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Oversampling using SMOTE.")

    elif mode == 'undersample':
        # 少数类别欠采样
        X_resampled = pd.DataFrame(X)
        y_resampled = pd.Series(y)

        # 获取主要类别和少数类别的样本数
        majority_class = y_resampled.value_counts().idxmax()
        minority_class = y_resampled.value_counts().idxmin()
        minority_count = y_resampled.value_counts().min()

        # 分离主要和少数样本
        X_majority = X_resampled[y_resampled == majority_class]
        X_minority = X_resampled[y_resampled == minority_class]

        # 下采样主要类别
        X_majority_downsampled, y_majority_downsampled = resample(
            X_majority, y_resampled[y_resampled == majority_class],
            replace=False, n_samples=minority_count, random_state=42
        )

        # 组合下采样的主要类别和少数类别
        X_resampled = pd.concat([X_majority_downsampled, X_minority])
        y_resampled = pd.concat([y_majority_downsampled, y_resampled[y_resampled == minority_class]])
        print("Performed undersampling.")

    elif mode == 'weighted':
        # 为加权损失函数计算类权重
        try:
            class_weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
            class_weights = {class_label: weight for class_label, weight in zip(unique_classes, class_weights_array)}
            print("Computed class weights for a weighted loss function.")
        except ValueError as e:
            print("Error in computing class weights:", e)
            print("Check if 'y' contains unexpected labels or missing values.")
            raise

        # 不进行重采样，返回原始数据
        X_resampled, y_resampled = X, y

    else:
        raise ValueError("Invalid mode. Choose 'oversample', 'undersample', or 'weighted'.")

    return X_resampled, y_resampled, class_weights