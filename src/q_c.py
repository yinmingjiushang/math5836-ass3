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
model_sel_path = "../results/q_c/best_model_results.csv"
# =========================

qc_1_path = "../out/q_c/visualizations"
qc_2_path = "../results/q_c"
results_file_path = f"{qc_2_path}/model_results.csv"
best_results_file_path = f"{qc_2_path}/best_model_results.csv"

if ed_state == 0:

    os.makedirs("../out/q_c/model_results", exist_ok=True)
    os.makedirs(qc_1_path, exist_ok=True)
    os.makedirs(qc_2_path, exist_ok=True)







def main():




if __name__ == "__main__":
    main()