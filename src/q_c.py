import os
import shutil
import math
import openml
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import export_graphviz, export_text
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

import q_b
import q_a

# define
# =========================
ed_state = 0
num_experiments = 5
# model train param
test_size = 0.6
random_seed = 42
model_sel_path = "../results/q_b/best_model_results.csv"
# 控制数据集选择模式：True 表示随机选择，False 表示固定选择
random_selection = False
specific_data_id = 19  # 固定数据集 ID，当 random_selection=False 时使用
# =========================


qc_1_path = "../out/q_c/visualizations"
qc_2_path = "../results/q_c"
results_file_path = f"{qc_2_path}/model_results.csv"
best_results_file_path = f"{qc_2_path}/best_model_results.csv"

if ed_state == 0:
    os.makedirs("../out/q_c/model_results", exist_ok=True)
    os.makedirs(qc_1_path, exist_ok=True)
    os.makedirs(qc_2_path, exist_ok=True)

# 字符串与函数的映射字典
model_mapping = {
    "Unpruned Decision Tree": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_basic_decision_tree(
        X, y, num_experiments, test_size, random_seed, output_dir),
    "Pruned Decision Tree": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_pruned_postprocessed_decision_tree(
        X, y, num_experiments, test_size, random_seed, output_dir),
    "Random Forest": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_model(
        RandomForestClassifier(n_estimators=100, random_state=random_seed), X, y, num_experiments, random_seed),
    "XGBoost": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_model(
        XGBClassifier(eval_metric='mlogloss', random_state=random_seed), X, y, num_experiments, random_seed),
    "Gradient Boosting": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_model(
        GradientBoostingClassifier(random_state=random_seed), X, y, num_experiments, random_seed),
    "Neural Network (Adam)": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_model(
        MLPClassifier(hidden_layer_sizes=(100,), solver='adam', max_iter=500, random_state=random_seed), X, y, num_experiments, random_seed),
    "Neural Network (SGD)": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.train_and_evaluate_model(
        MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', max_iter=500, random_state=random_seed), X, y, num_experiments, random_seed),
    "L2 vs Dropout": lambda X, y, num_experiments, test_size, random_seed, output_dir: q_a.compare_l2_and_dropout(
        X, y, num_experiments, test_size, random_seed, output_dir)
}


def save_results_to_csv(results, output_file):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


def get_small_datasets(max_samples=10000):
    datasets = openml.datasets.list_datasets(output_format="dataframe")
    # 筛选出样本数小于 max_samples 的分类任务数据集
    small_classification_datasets = datasets[(datasets['NumberOfInstances'] <= max_samples) &
                                             (datasets['NumberOfInstances'] > 0) &
                                             (datasets['NumberOfClasses'] > 1)]
    return small_classification_datasets[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']]


def data_input():
    if random_selection:
        # 随机选择数据集
        small_datasets = get_small_datasets()
        random_id = small_datasets.sample(1).iloc[0]['did']
        print(f"Selected Dataset ID: {random_id}")
    else:
        # 使用特定数据集 ID
        random_id = specific_data_id
        print(f"Using Specific Dataset ID: {random_id}")

    # 尝试加载数据集
    dataset = fetch_openml(data_id=random_id, as_frame=False)
    data, target = dataset.data, dataset.target

    # 将数据转换为 DataFrame 格式
    X = pd.DataFrame(data, columns=dataset.feature_names)
    y = pd.Series(target, name="target")

    # 调用预处理函数
    X, y = preprocess_data(X, y)

    return X, y

def preprocess_data(X, y):
    # 将所有非数值列转换为数值
    for column in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # 检查并转换 y 是否为数值类型
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # 仅填充数值列的缺失值
    X[X.select_dtypes(include=['float64', 'int64']).columns] = X.select_dtypes(include=['float64', 'int64']).fillna(X.mean())

    return X, y

def save_to_csv(X, y, output_path):
    # 如果 y 是 ndarray 类型，将其转换为 Series
    if isinstance(y, (list, np.ndarray)):
        y = pd.Series(y, name="target")

    # 将 X 和 y 合并为一个 DataFrame
    combined_df = pd.concat([X, y], axis=1)

    # 保存为 CSV 文件
    combined_df.to_csv(output_path, index=False)

def analyze_and_visualize_data(X, y, output_dir):

    # 可视化 `y` 的分布
    def visualize_target_distribution(y):
        y_1d = y.to_numpy().ravel() if isinstance(y, pd.DataFrame) else y  # 确保 y 是一维数组
        y_series = pd.Series(y_1d)  # 转换为 pandas Series 以便绘图
        plt.figure(figsize=(8, 6))
        if y_series.nunique() > 10:  # 若 y 是连续变量
            sns.histplot(y_series, kde=True, color='skyblue', edgecolor='black')
            plt.title('Target Variable Distribution (Continuous)')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
        else:  # 若 y 是分类变量
            sns.countplot(x=y_series, hue=y_series, palette='viridis', legend=False)
            plt.title('Target Variable Distribution (Categorical)')
            plt.xlabel('Target Class')
            plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution.png")
        plt.close()

    # 可视化每个特征的分布
    def visualize_feature_distribution(X):
        # 计算特征数量
        num_features = X.shape[1]

        # 动态计算行数和列数，使其足够大
        num_cols = 3  # 固定列数
        num_rows = math.ceil(num_features / num_cols)  # 计算所需行数

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 绘制直方图
        plt.figure(figsize=(15, 10))
        X.hist(bins=20, figsize=(15, 10), layout=(num_rows, num_cols), color='skyblue', edgecolor='black')
        plt.suptitle('Feature Distributions')
        plt.tight_layout()

        # 保存图像
        plt.savefig(f"{output_dir}/feature_distribution.png")
        plt.close()

    # 特征和目标值的关系（配对图）
    def visualize_pairplot(X, y):
        pairplot_data = X.copy()
        pairplot_data['Target'] = y.to_numpy().ravel() if isinstance(y, pd.DataFrame) else y  # 保持兼容性
        pairplot_fig = sns.pairplot(pairplot_data, hue='Target', palette='husl', plot_kws={'alpha': 0.5})
        pairplot_fig.fig.tight_layout()
        pairplot_fig.savefig(f"{output_dir}/pairplot.png")
        plt.close()

    # 相关性热图
    def visualize_correlation_heatmap(X):
        plt.figure(figsize=(10, 8))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    # 调用可视化函数
    visualize_target_distribution(y)
    visualize_feature_distribution(X)
    visualize_correlation_heatmap(X)

    # 生成和保存特征的统计摘要
    summary_stats = X.describe()
    print("Summary Statistics for Each Feature:\n", summary_stats)
    summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')


def model_selection(model_sel_path, out_path):
    df = pd.read_csv(model_sel_path)

    top_2_models = df.sort_values(by='Best Accuracy', ascending=False).head(2)
    print(top_2_models)
    top_2_models.to_csv(f'{out_path}/model_sel.csv', index=False)

    model1_name, model2_name = top_2_models['Model'].iloc[0], top_2_models['Model'].iloc[1]

    return model1_name, model2_name



def main():
    results = []
    best_results = []

    X, y = data_input()

    save_to_csv(X, y, "../out/q_c/data.csv")

    analyze_and_visualize_data(X, y, qc_1_path)

    model1_name, model2_name = model_selection(model_sel_path, qc_2_path)

    model_names = [model1_name, model2_name]

    # 遍历每个模型并计算结果
    for model_name in model_names:
        # 调用 model_mapping 中对应的模型函数并解包返回的结果
        (mean_accuracy, std_accuracy, var_accuracy,
         mean_auc, std_auc, var_auc,
         mean_f1, std_f1, var_f1,
         best_accuracy, best_auc, best_f1) = model_mapping[model_name](
            X, y, num_experiments, test_size, random_seed, qc_2_path
        )

        # 将均值和方差等统计信息添加到 results 列表
        results.append({
            "Model": model_name,
            "Mean Accuracy": mean_accuracy,
            "Variance Accuracy": var_accuracy,
            "Std Accuracy": std_accuracy,
            "Mean AUC": mean_auc,
            "Variance AUC": var_auc,
            "Std AUC": std_auc,
            "Mean F1 Score": mean_f1,
            "Variance F1 Score": var_f1,
            "Std F1 Score": std_f1
        })

        # 将最佳结果添加到 best_results 列表
        best_results.append({
            "Model": model_name,
            "Best Accuracy": best_accuracy,
            "Best AUC": best_auc,
            "Best F1 Score": best_f1
        })

    # Save all results to CSV
    save_results_to_csv(results, results_file_path)
    save_results_to_csv(best_results, best_results_file_path)

if __name__ == "__main__":
    main()
