import os
import shutil

from ucimlrepo import fetch_ucirepo
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

import q_a

# define
# =========================
ed_state = 0
num_experiments = 5
# model train param
test_size = 0.6
random_seed = 42
model_sel_path = "../results/q_a/best_model_results.csv"
# =========================

qb_1_path = "../out/q_b/visualizations"
qb_2_path = "../results/q_b"
results_file_path = f"{qb_2_path}/model_results.csv"
best_results_file_path = f"{qb_2_path}/best_model_results.csv"

if ed_state == 0:

    os.makedirs("../out/q_b/model_results", exist_ok=True)
    os.makedirs(qb_1_path, exist_ok=True)
    os.makedirs(qb_2_path, exist_ok=True)

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


def data_input():
    # fetch dataset
    contraceptive_method_choice = fetch_ucirepo(id=30)

    # data (as pandas dataframes)
    X = contraceptive_method_choice.data.features
    y = contraceptive_method_choice.data.targets
    print("y 的维度:", y.shape)
    # metadata
    print(contraceptive_method_choice.metadata)

    # variable information
    print(contraceptive_method_choice.variables)

    return X, y


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

    model1_name, model2_name = model_selection(model_sel_path, qb_2_path)

    model_names = [model1_name, model2_name]

    # 遍历每个模型并计算结果
    for model_name in model_names:
        # 调用 model_mapping 中对应的模型函数并解包返回的结果
        (mean_accuracy, std_accuracy, var_accuracy,
         mean_auc, std_auc, var_auc,
         mean_f1, std_f1, var_f1,
         best_accuracy, best_auc, best_f1) = model_mapping[model_name](
            X, y, num_experiments, test_size, random_seed, qb_2_path
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
