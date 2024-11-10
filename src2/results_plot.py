import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import out_path

def best_model_results(file_path,out_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Plotting the line graph for different metrics
    plt.figure(figsize=(10, 6))

    # Defining distinct colors for each line
    colors = ['blue', 'green', 'red']

    # Plotting each metric as a separate line with specified colors
    for i, metric in enumerate(['Best Accuracy', 'Best AUC', 'Best F1 Score']):
        plt.plot(df['Model'], df[metric], label=metric, marker='o', color=colors[i])

    # Adding title and labels
    plt.title('Performance Metrics for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{out_path}/best_model_results.png")
    plt.close()



def plot_average_performance(file_path, output_path):

    df = pd.read_csv(file_path)
    # Plotting the average performance metrics
    plt.figure(figsize=(12, 6))

    # Plotting Mean Accuracy, Mean AUC, and Mean F1 Score
    for metric in ['Mean Accuracy', 'Mean AUC', 'Mean F1 Score']:
        plt.plot(df['Model'], df[metric], label=metric, marker='o')

    plt.title('Average Performance Metrics for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/average_performance.png")
    plt.close()

def plot_stability(file_path, output_path):
    df = pd.read_csv(file_path)
    # Plotting the stability of the models
    plt.figure(figsize=(12, 6))

    # Plotting Std Accuracy, Std AUC, and Std F1 Score
    for metric in ['Std Accuracy', 'Std AUC', 'Std F1 Score']:
        plt.plot(df['Model'], df[metric], label=metric, marker='o')

    plt.title('Stability of Different Models (Standard Deviation)')
    plt.xlabel('Model')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/stability.png")
    plt.close()

def sel_best_model_results(file_path, out_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # 使用更柔和的浅色调
    colors = ['#aec6cf', '#b0e0a8', '#f9c6cb']  # 淡灰蓝、淡灰绿、粉灰色

    # 设置图形和柱宽
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(3)  # Since we have 3 metrics: Accuracy, AUC, F1 Score

    # Extracting values for each model to ensure grouping by metrics
    model_1 = df.iloc[0, 1:].values  # First model's metrics
    model_2 = df.iloc[1, 1:].values  # Second model's metrics

    # Plotting grouped bars for each metric with softer colors
    plt.bar(index, model_1, bar_width, color=colors[0], label=df['Model'][0])
    plt.bar(index + bar_width, model_2, bar_width, color=colors[1], label=df['Model'][1])

    # Adding titles, labels, and legend
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Comparison by Metrics for Each Model')
    plt.xticks(index + bar_width / 2, ['Best Accuracy', 'Best AUC', 'Best F1 Score'])
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{out_path}/best_model_results.png")
    plt.close()


def sel_plot_average_performance(file_path, output_path):
    df = pd.read_csv(file_path)
    # 使用更柔和的浅色调
    colors = ['#aec6cf', '#b0e0a8', '#f9c6cb']

    # 设置图形大小
    plt.figure(figsize=(10, 6))
    bar_width = 0.3
    index = np.arange(len(df['Model']))

    # 绘制平均性能的柱状图
    plt.bar(index, df['Mean Accuracy'], bar_width, color=colors[0], label='Mean Accuracy')
    plt.bar(index + bar_width, df['Mean AUC'], bar_width, color=colors[1], label='Mean AUC')
    plt.bar(index + 2 * bar_width, df['Mean F1 Score'], bar_width, color=colors[2], label='Mean F1 Score')

    # 添加标题和标签
    plt.title('Average Performance Metrics for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(index + bar_width, df['Model'], rotation=45)
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{output_path}/average_performance.png")
    plt.close()


def sel_plot_stability(file_path, output_path):
    df = pd.read_csv(file_path)
    # 使用更柔和的浅色调
    colors = ['#aec6cf', '#b0e0a8', '#f9c6cb']

    # 设置图形大小
    plt.figure(figsize=(10, 6))
    bar_width = 0.3
    index = np.arange(len(df['Model']))

    # 绘制稳定性的柱状图
    plt.bar(index, df['Std Accuracy'], bar_width, color=colors[0], label='Std Accuracy')
    plt.bar(index + bar_width, df['Std AUC'], bar_width, color=colors[1], label='Std AUC')
    plt.bar(index + 2 * bar_width, df['Std F1 Score'], bar_width, color=colors[2], label='Std F1 Score')

    # 添加标题和标签
    plt.title('Stability of Different Models (Standard Deviation)')
    plt.xlabel('Model')
    plt.ylabel('Standard Deviation')
    plt.xticks(index + bar_width, df['Model'], rotation=45)
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{output_path}/stability.png")
    plt.close()


def results_plot(flag):
    if flag == "q_a":
        best_model_results(f"{out_path.OutPath.results_path(flag)}/best_results.csv",f"{out_path.OutPath.analysis_path(flag)}")
        plot_average_performance(f"{out_path.OutPath.results_path(flag)}/results.csv", f"{out_path.OutPath.analysis_path(flag)}")
        plot_stability(f"{out_path.OutPath.results_path(flag)}/results.csv", f"{out_path.OutPath.analysis_path(flag)}")
    if flag == "q_b" or flag == "q_c":
        sel_best_model_results(f"{out_path.OutPath.results_path(flag)}/best_results.csv", f"{out_path.OutPath.analysis_path(flag)}")
        sel_plot_average_performance(f"{out_path.OutPath.results_path(flag)}/results.csv", f"{out_path.OutPath.analysis_path(flag)}")
        sel_plot_stability(f"{out_path.OutPath.results_path(flag)}/results.csv", f"{out_path.OutPath.analysis_path(flag)}")


