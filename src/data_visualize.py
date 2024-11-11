import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import out_path

def visualize_data(flag, X, y):
    output_dir = out_path.OutPath.visualization_path(flag)

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
    visualize_pairplot(X, y)
    visualize_correlation_heatmap(X)

    # 生成和保存特征的统计摘要
    summary_stats = X.describe()
    print("Summary Statistics for Each Feature:\n", summary_stats)
    summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
